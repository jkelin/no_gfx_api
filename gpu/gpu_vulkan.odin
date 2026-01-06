
package gpu

import "core:slice"
import "core:log"
import "base:runtime"
import vmem "core:mem/virtual"
import "core:mem"
import rbt "core:container/rbtree"

import sdl "vendor:sdl3"
import vk "vendor:vulkan"
import "vma"

Push_Constant_Size :: size_of(rawptr) * 4  // Max: vert_data, frag_data, vert_indirect_data, frag_indirect_data
Max_Textures :: 65535

@(private="file")
GPU_Alloc_Meta :: struct #all_or_none
{
    buf_handle: vk.Buffer,
    allocation: vma.Allocation,
    device_address: vk.DeviceAddress,
    align: u32,
    buf_size: vk.DeviceSize,
}

@(private="file")
Alloc_Range :: struct
{
    ptr: u64,
    size: u32,
}

@(private="file")
Timeline :: struct
{
    sem: vk.Semaphore,
    val: u64,
    recording: bool,
}

Image_View_Info :: struct
{
    info: vk.ImageViewCreateInfo,
    view: vk.ImageView,
}

Sampler_Info :: struct
{
    info: vk.SamplerCreateInfo,
    sampler: vk.Sampler,
}

@(private="file")
Context :: struct
{
    instance: vk.Instance,
    debug_messenger: vk.DebugUtilsMessengerEXT,
    surface: vk.SurfaceKHR,

    vma_allocator: vma.Allocator,

    gpu_allocs: [dynamic]GPU_Alloc_Meta,
    // TODO: freelist of gpu allocs
    cpu_ptr_to_alloc: map[rawptr]u32,  // Each entry has an index to its corresponding GPU allocation
    gpu_ptr_to_alloc: map[rawptr]u32,  // From base GPU allocation pointer to metadata
    alloc_tree: rbt.Tree(Alloc_Range, u32),

    phys_device: vk.PhysicalDevice,
    device: vk.Device,
    queue: vk.Queue,
    queue_family_idx: u32,

    cmd_pool: vk.CommandPool,
    cmd_bufs: [10]vk.CommandBuffer,
    cmd_bufs_timelines: [10]Timeline,

    // Common resources
    textures_desc_layout: vk.DescriptorSetLayout,
    textures_rw_desc_layout: vk.DescriptorSetLayout,
    samplers_desc_layout: vk.DescriptorSetLayout,
    data_desc_layout: vk.DescriptorSetLayout,
    indirect_data_desc_layout: vk.DescriptorSetLayout,
    common_pipeline_layout: vk.PipelineLayout,

    // Descriptor objects
    image_views: map[vk.Image][dynamic]Image_View_Info,
    samplers: [dynamic]Sampler_Info,

    // Swapchain
    swapchain: Swapchain,
    swapchain_image_idx: u32,

    // Descriptor sizes
    texture_desc_size: u32,
    texture_rw_desc_size: u32,
    sampler_desc_size: u32,
}

// Initialization

@(private="file")
ctx: Context
@(private="file")
vk_logger: log.Logger

_init :: proc(window: ^sdl.Window, frames_in_flight: u32)
{
    init_scratch_arenas()

    scratch, _ := acquire_scratch()

    // Load vulkan function pointers
    vk.load_proc_addresses_global(cast(rawptr) sdl.Vulkan_GetVkGetInstanceProcAddr())

    vk_logger = context.logger

    // Create instance
    {
        when ODIN_DEBUG
        {
            layers := []cstring {
                "VK_LAYER_KHRONOS_validation",
            }
        }
        else
        {
            layers := []cstring {}
        }

        count: u32
        instance_extensions := sdl.Vulkan_GetInstanceExtensions(&count)
        extensions := slice.concatenate([][]cstring {
            instance_extensions[:count],
            {
                vk.EXT_DEBUG_UTILS_EXTENSION_NAME,
            }
        }, allocator = scratch)

        debug_messenger_ci := vk.DebugUtilsMessengerCreateInfoEXT {
            sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            messageSeverity = { .WARNING, .ERROR },
            messageType = { .VALIDATION, .PERFORMANCE },
            pfnUserCallback = vk_debug_callback
        }

        when ODIN_DEBUG
        {
            validation_features := []vk.ValidationFeatureEnableEXT {
                // .GPU_ASSISTED,
                // .GPU_ASSISTED_RESERVE_BINDING_SLOT,
                .SYNCHRONIZATION_VALIDATION,
            }
        }
        else
        {
            validation_features := []vk.ValidationFeatureEnableEXT {}
        }

        next: rawptr
        next = &debug_messenger_ci
        next = &vk.ValidationFeaturesEXT {
            sType = .VALIDATION_FEATURES_EXT,
            pNext = next,
            enabledValidationFeatureCount = u32(len(validation_features)),
            pEnabledValidationFeatures = raw_data(validation_features),
        }

        vk_check(vk.CreateInstance(&{
            sType = .INSTANCE_CREATE_INFO,
            pApplicationInfo = &{
                sType = .APPLICATION_INFO,
                apiVersion = vk.API_VERSION_1_3,
            },
            enabledLayerCount = u32(len(layers)),
            ppEnabledLayerNames = raw_data(layers),
            enabledExtensionCount = u32(len(extensions)),
            ppEnabledExtensionNames = raw_data(extensions),
            pNext = next,
        }, nil, &ctx.instance))

        vk.load_proc_addresses_instance(ctx.instance)
        assert(vk.DestroyInstance != nil, "Failed to load Vulkan instance API")

        vk_check(vk.CreateDebugUtilsMessengerEXT(ctx.instance, &debug_messenger_ci, nil, &ctx.debug_messenger))
    }

    // Create surface
    {
        ok_s := sdl.Vulkan_CreateSurface(window, ctx.instance, nil, &ctx.surface)
        if !ok_s do fatal_error("Could not create vulkan surface.")
    }

    // Physical device
    phys_device_count: u32
    vk_check(vk.EnumeratePhysicalDevices(ctx.instance, &phys_device_count, nil))
    if phys_device_count == 0 do fatal_error("Did not find any GPUs!")
    phys_devices := make([]vk.PhysicalDevice, phys_device_count, allocator = scratch)
    vk_check(vk.EnumeratePhysicalDevices(ctx.instance, &phys_device_count, raw_data(phys_devices)))

    chosen_phys_device: vk.PhysicalDevice
    queue_family_idx: u32
    found := false
    device_loop: for candidate in phys_devices
    {
        queue_family_count: u32
        vk.GetPhysicalDeviceQueueFamilyProperties(candidate, &queue_family_count, nil)
        queue_families := make([]vk.QueueFamilyProperties, queue_family_count, allocator = scratch)
        vk.GetPhysicalDeviceQueueFamilyProperties(candidate, &queue_family_count, raw_data(queue_families))

        for family, i in queue_families
        {
            supports_graphics := .GRAPHICS in family.queueFlags
            supports_present: b32
            vk_check(vk.GetPhysicalDeviceSurfaceSupportKHR(candidate, u32(i), ctx.surface, &supports_present))

            if supports_graphics && supports_present
            {
                chosen_phys_device = candidate
                queue_family_idx = u32(i)
                found = true
                break device_loop
            }
        }
    }

    if !found do fatal_error("Could not find suitable GPU.")

    ctx.phys_device = chosen_phys_device

    // Check descriptor sizes
    props := vk.PhysicalDeviceDescriptorBufferPropertiesEXT {
        sType = .PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT
    };
    props2 := vk.PhysicalDeviceProperties2 {
        sType = .PHYSICAL_DEVICE_PROPERTIES_2,
        pNext = &props
    };
    vk.GetPhysicalDeviceProperties2(ctx.phys_device, &props2)
    ensure(props.storageImageDescriptorSize <= 32, "Unexpected storage image descriptor size.")
    ensure(props.sampledImageDescriptorSize <= 32, "Unexpected sampled texture descriptor size.")
    ensure(props.samplerDescriptorSize <= 16, "Unexpected sampler descriptor size.")
    ctx.texture_desc_size = u32(props.sampledImageDescriptorSize)
    ctx.texture_rw_desc_size = u32(props.storageImageDescriptorSize)
    ctx.sampler_desc_size = u32(props.samplerDescriptorSize)

    queue_priorities := []f32 { 0.0, 1.0 }
    queue_create_infos := []vk.DeviceQueueCreateInfo {
        {
            sType = .DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex = queue_family_idx,
            queueCount = u32(len(queue_priorities)),
            pQueuePriorities = raw_data(queue_priorities),
        },
    }

    // Device
    device_extensions := []cstring {
        vk.KHR_SWAPCHAIN_EXTENSION_NAME,
        vk.EXT_SHADER_OBJECT_EXTENSION_NAME,
        vk.EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
        vk.KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME,
    }

    next: rawptr
    next = &vk.PhysicalDeviceVulkan12Features {
        sType = .PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        pNext = next,
        runtimeDescriptorArray = true,
        shaderSampledImageArrayNonUniformIndexing = true,
        timelineSemaphore = true,
        bufferDeviceAddress = true,
        drawIndirectCount = true,
        scalarBlockLayout = true,
    }
    next = &vk.PhysicalDeviceVulkan11Features {
        sType = .PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        pNext = next,
        shaderDrawParameters = true,
    }
    next = &vk.PhysicalDeviceVulkan13Features {
        sType = .PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        pNext = next,
        dynamicRendering = true,
        synchronization2 = true,
    }
    next = &vk.PhysicalDeviceDescriptorBufferFeaturesEXT {
        sType = .PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
        pNext = next,
        descriptorBuffer = true,
    }
    next = &vk.PhysicalDeviceShaderObjectFeaturesEXT {
        sType = .PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
        pNext = next,
        shaderObject = true,
    }
    next = &vk.PhysicalDeviceDepthClipEnableFeaturesEXT {
        sType = .PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT,
        pNext = next,
        depthClipEnable = true,
    }
    next = &vk.PhysicalDeviceFeatures2 {
        sType = .PHYSICAL_DEVICE_FEATURES_2,
        pNext = next,
        features = {
            shaderInt64 = true,
            vertexPipelineStoresAndAtomics = true,
        }
    }

    device_ci := vk.DeviceCreateInfo {
        sType = .DEVICE_CREATE_INFO,
        pNext = next,
        queueCreateInfoCount = u32(len(queue_create_infos)),
        pQueueCreateInfos = raw_data(queue_create_infos),
        enabledExtensionCount = u32(len(device_extensions)),
        ppEnabledExtensionNames = raw_data(device_extensions),
    }
    vk_check(vk.CreateDevice(chosen_phys_device, &device_ci, nil, &ctx.device))

    vk.load_proc_addresses_device(ctx.device)
    if vk.BeginCommandBuffer == nil do fatal_error("Failed to load Vulkan device API")

    vk.GetDeviceQueue(ctx.device, queue_family_idx, 0, &ctx.queue)

    // Command buffers
    cmd_pool_ci := vk.CommandPoolCreateInfo {
        sType = .COMMAND_POOL_CREATE_INFO,
        queueFamilyIndex = ctx.queue_family_idx,
        flags = { .TRANSIENT, .RESET_COMMAND_BUFFER }
    }
    vk_check(vk.CreateCommandPool(ctx.device, &cmd_pool_ci, nil, &ctx.cmd_pool))

    cmd_buf_ai := vk.CommandBufferAllocateInfo {
        sType = .COMMAND_BUFFER_ALLOCATE_INFO,
        commandPool = ctx.cmd_pool,
        level = .PRIMARY,
        commandBufferCount = len(ctx.cmd_bufs),
    }
    vk_check(vk.AllocateCommandBuffers(ctx.device, &cmd_buf_ai, &ctx.cmd_bufs[0]))

    for &timeline in ctx.cmd_bufs_timelines
    {
        next_sem: rawptr
        next_sem = &vk.SemaphoreTypeCreateInfo {
            sType = .SEMAPHORE_TYPE_CREATE_INFO,
            pNext = next_sem,
            semaphoreType = .TIMELINE,
            initialValue = 0,
        }
        sem_ci := vk.SemaphoreCreateInfo {
            sType = .SEMAPHORE_CREATE_INFO,
            pNext = next_sem
        }
        vk_check(vk.CreateSemaphore(ctx.device, &sem_ci, nil, &timeline.sem))
    }

    // Common resources
    {
        push_constant_ranges := []vk.PushConstantRange {
            {
                stageFlags = { .VERTEX, .FRAGMENT },
                size = Push_Constant_Size,
            }
        }

        {
            layout_ci := vk.DescriptorSetLayoutCreateInfo {
                sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                flags = { .DESCRIPTOR_BUFFER_EXT },
                bindingCount = 1,
                pBindings = &vk.DescriptorSetLayoutBinding {
                    binding = 0,
                    descriptorType = .SAMPLED_IMAGE,
                    descriptorCount = Max_Textures,
                    stageFlags = { .VERTEX, .FRAGMENT, .COMPUTE },
                },
            }
            vk_check(vk.CreateDescriptorSetLayout(ctx.device, &layout_ci, nil, &ctx.textures_desc_layout))
        }
        {
            layout_ci := vk.DescriptorSetLayoutCreateInfo {
                sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                flags = { .DESCRIPTOR_BUFFER_EXT },
                bindingCount = 1,
                pBindings = &vk.DescriptorSetLayoutBinding {
                    binding = 0,
                    descriptorType = .STORAGE_IMAGE,
                    descriptorCount = Max_Textures,
                    stageFlags = { .VERTEX, .FRAGMENT, .COMPUTE },
                },
            }
            vk_check(vk.CreateDescriptorSetLayout(ctx.device, &layout_ci, nil, &ctx.textures_rw_desc_layout))
        }
        {
            layout_ci := vk.DescriptorSetLayoutCreateInfo {
                sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                flags = { .DESCRIPTOR_BUFFER_EXT },
                bindingCount = 1,
                pBindings = &vk.DescriptorSetLayoutBinding {
                    binding = 0,
                    descriptorType = .SAMPLER,
                    descriptorCount = Max_Textures,
                    stageFlags = { .VERTEX, .FRAGMENT, .COMPUTE },
                },
            }
            vk_check(vk.CreateDescriptorSetLayout(ctx.device, &layout_ci, nil, &ctx.samplers_desc_layout))
        }
        {
            layout_ci := vk.DescriptorSetLayoutCreateInfo {
                sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                flags = { .DESCRIPTOR_BUFFER_EXT },
                bindingCount = 1,
                pBindings = &vk.DescriptorSetLayoutBinding {
                    binding = 0,
                    descriptorType = .STORAGE_BUFFER,
                    descriptorCount = 1,
                    stageFlags = { .VERTEX, .FRAGMENT },
                },
            }
            vk_check(vk.CreateDescriptorSetLayout(ctx.device, &layout_ci, nil, &ctx.data_desc_layout))
        }
        {
            layout_ci := vk.DescriptorSetLayoutCreateInfo {
                sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                flags = { .DESCRIPTOR_BUFFER_EXT },
                bindingCount = 1,
                pBindings = &vk.DescriptorSetLayoutBinding {
                    binding = 0,
                    descriptorType = .STORAGE_BUFFER,
                    descriptorCount = 1,
                    stageFlags = { .VERTEX, .FRAGMENT },
                },
            }
            vk_check(vk.CreateDescriptorSetLayout(ctx.device, &layout_ci, nil, &ctx.indirect_data_desc_layout))
        }

        desc_layouts := []vk.DescriptorSetLayout {
            ctx.textures_desc_layout,
            ctx.textures_rw_desc_layout,
            ctx.samplers_desc_layout,
            ctx.data_desc_layout,
            ctx.indirect_data_desc_layout,
        }
        pipeline_layout_ci := vk.PipelineLayoutCreateInfo {
            sType = .PIPELINE_LAYOUT_CREATE_INFO,
            pushConstantRangeCount = u32(len(push_constant_ranges)),
            pPushConstantRanges = raw_data(push_constant_ranges),
            setLayoutCount = u32(len(desc_layouts)),
            pSetLayouts = raw_data(desc_layouts),
        }
        vk_check(vk.CreatePipelineLayout(ctx.device, &pipeline_layout_ci, nil, &ctx.common_pipeline_layout))

        win_width, win_height: i32
        assert(sdl.GetWindowSize(window, &win_width, &win_height))
        ctx.swapchain = create_swapchain(u32(win_width), u32(win_height), frames_in_flight)
    }

    // Tree init
    rbt.init_cmp(&ctx.alloc_tree, proc(range_a: Alloc_Range, range_b: Alloc_Range) -> rbt.Ordering {
        // NOTE: When searching, Alloc_Range { ptr, 0 } is used.
        diff_ba := int(range_b.ptr) - int(range_a.ptr)
        diff_ab := int(range_a.ptr) - int(range_b.ptr)
        if diff_ba >= 0 && diff_ba < int(range_a.size) {
            return .Equal
        } else if diff_ab >= 0 && diff_ab < int(range_b.size) {
            return .Equal
        } else if range_a.ptr < range_b.ptr {
            return .Less
        } else {
            return .Greater
        }
    })

    // VMA allocator
    vma_vulkan_procs := vma.create_vulkan_functions()
    ok_vma := vma.create_allocator({
        flags = { .Buffer_Device_Address },
        instance = ctx.instance,
        vulkan_api_version = 1003000,  // 1.3
        physical_device = ctx.phys_device,
        device = ctx.device,
        vulkan_functions = &vma_vulkan_procs,
    }, &ctx.vma_allocator)
    assert(ok_vma == .SUCCESS)
}

_cleanup :: proc()
{
    vk.DestroyCommandPool(ctx.device, ctx.cmd_pool, nil)

    for &sampler in ctx.samplers {
        vk.DestroySampler(ctx.device, sampler.sampler, nil)
    }

    destroy_swapchain(ctx.swapchain)
    for timeline in ctx.cmd_bufs_timelines {
        vk.DestroySemaphore(ctx.device, timeline.sem, nil)
    }

    vk.DestroyDescriptorSetLayout(ctx.device, ctx.textures_desc_layout, nil)
    vk.DestroyDescriptorSetLayout(ctx.device, ctx.textures_rw_desc_layout, nil)
    vk.DestroyDescriptorSetLayout(ctx.device, ctx.samplers_desc_layout, nil)
    vk.DestroyDescriptorSetLayout(ctx.device, ctx.data_desc_layout, nil)
    vk.DestroyDescriptorSetLayout(ctx.device, ctx.indirect_data_desc_layout, nil)
    vk.DestroyPipelineLayout(ctx.device, ctx.common_pipeline_layout, nil)

    vma.destroy_allocator(ctx.vma_allocator)

    vk.DestroyDevice(ctx.device, nil)
}

_wait_idle :: proc()
{
    vk.DeviceWaitIdle(ctx.device)
}

_swapchain_acquire_next :: proc() -> Texture
{
    fence_ci := vk.FenceCreateInfo { sType = .FENCE_CREATE_INFO }
    fence: vk.Fence
    vk_check(vk.CreateFence(ctx.device, &fence_ci, nil, &fence))
    defer vk.DestroyFence(ctx.device, fence, nil)

    vk_check(vk.AcquireNextImageKHR(ctx.device, ctx.swapchain.handle, max(u64), {}, fence, &ctx.swapchain_image_idx))

    vk_check(vk.WaitForFences(ctx.device, 1, &fence, true, max(u64)))

    // Transition layout from swapchain
    {
        cmd_buf := vk_acquire_cmd_buf(ctx.queue)

        cmd_buf_bi := vk.CommandBufferBeginInfo {
            sType = .COMMAND_BUFFER_BEGIN_INFO,
            flags = { .ONE_TIME_SUBMIT },
        }
        vk_check(vk.BeginCommandBuffer(cmd_buf, &cmd_buf_bi))

        transition := vk.ImageMemoryBarrier2 {
            sType = .IMAGE_MEMORY_BARRIER_2,
            image = ctx.swapchain.images[ctx.swapchain_image_idx],
            subresourceRange = {
                aspectMask = { .COLOR },
                levelCount = 1,
                layerCount = 1,
            },
            oldLayout = .UNDEFINED,
            newLayout = .GENERAL,
            srcStageMask = { .ALL_COMMANDS },
            srcAccessMask = { .MEMORY_WRITE },
            dstStageMask = { .COLOR_ATTACHMENT_OUTPUT },
            dstAccessMask = { .MEMORY_READ, .MEMORY_WRITE },
        }
        vk.CmdPipelineBarrier2(cmd_buf, &vk.DependencyInfo {
            sType = .DEPENDENCY_INFO,
            imageMemoryBarrierCount = 1,
            pImageMemoryBarriers = &transition,
        })

        vk_check(vk.EndCommandBuffer(cmd_buf))

        vk_submit_cmd_buf(ctx.queue, cmd_buf)
    }

    return Texture {
        dimensions = { ctx.swapchain.width, ctx.swapchain.height, 1 },
        format = .BGRA8_Unorm,
        handle = transmute(Texture_Handle) ctx.swapchain.images[ctx.swapchain_image_idx],
    }
}

_swapchain_present :: proc(queue: Queue, sem_wait: Semaphore, wait_value: u64)
{
    vk_queue := cast(vk.Queue) queue
    vk_sem_wait := transmute(vk.Semaphore) sem_wait

    present_semaphore := ctx.swapchain.present_semaphores[ctx.swapchain_image_idx]

    // NOTE: Workaround for the fact that swapchain presentation
    // only supports binary semaphores.
    // wait on sem_wait on wait_value and signal ctx.binary_sem
    {
        queue := ctx.queue

        // Switch to optimal layout for presentation (this is mandatory)
        cmd_buf: vk.CommandBuffer
        {
            cmd_buf = vk_acquire_cmd_buf(vk_queue)

            cmd_buf_bi := vk.CommandBufferBeginInfo {
                sType = .COMMAND_BUFFER_BEGIN_INFO,
                flags = { .ONE_TIME_SUBMIT },
            }
            vk_check(vk.BeginCommandBuffer(cmd_buf, &cmd_buf_bi))

            transition := vk.ImageMemoryBarrier2 {
                sType = .IMAGE_MEMORY_BARRIER_2,
                image = ctx.swapchain.images[ctx.swapchain_image_idx],
                subresourceRange = {
                    aspectMask = { .COLOR },
                    levelCount = 1,
                    layerCount = 1,
                },
                oldLayout = .GENERAL,
                newLayout = .PRESENT_SRC_KHR,
                srcStageMask = { .ALL_COMMANDS },
                srcAccessMask = { .MEMORY_WRITE },
                dstStageMask = { .COLOR_ATTACHMENT_OUTPUT },
                dstAccessMask = { .MEMORY_READ },
            }
            vk.CmdPipelineBarrier2(cmd_buf, &vk.DependencyInfo {
                sType = .DEPENDENCY_INFO,
                imageMemoryBarrierCount = 1,
                pImageMemoryBarriers = &transition,
            })

            vk_check(vk.EndCommandBuffer(cmd_buf))
        }

        timeline := vk_get_cmd_buf_timeline(vk_queue, cmd_buf)
        timeline.val += 1

        wait_stage_flags := vk.PipelineStageFlags { .COLOR_ATTACHMENT_OUTPUT }
        next: rawptr
        next = &vk.TimelineSemaphoreSubmitInfo {
            sType = .TIMELINE_SEMAPHORE_SUBMIT_INFO,
            pNext = next,
            waitSemaphoreValueCount = 1,
            pWaitSemaphoreValues = raw_data([]u64 {
                wait_value,
            }),
            signalSemaphoreValueCount = 2,
            pSignalSemaphoreValues = raw_data([]u64 {
                {},
                timeline.val,
            })
        }
        submit_info := vk.SubmitInfo {
            sType = .SUBMIT_INFO,
            pNext = next,
            commandBufferCount = 1,
            pCommandBuffers = &cmd_buf,
            waitSemaphoreCount = 1,
            pWaitSemaphores = raw_data([]vk.Semaphore {
                vk_sem_wait,
            }),
            pWaitDstStageMask = raw_data([]vk.PipelineStageFlags {
                wait_stage_flags,
            }),
            signalSemaphoreCount = 2,
            pSignalSemaphores = raw_data([]vk.Semaphore {
                present_semaphore,
                timeline.sem,
            }),
        }
        vk_check(vk.QueueSubmit(queue, 1, &submit_info, {}))

        timeline.recording = false
    }

    vk_check(vk.QueuePresentKHR(ctx.queue, &{
        sType = .PRESENT_INFO_KHR,
        swapchainCount = 1,
        waitSemaphoreCount = 1,
        pWaitSemaphores = &present_semaphore,
        pSwapchains = &ctx.swapchain.handle,
        pImageIndices = &ctx.swapchain_image_idx,
    }))
}

// Memory

_mem_alloc :: proc(bytes: u64, align: u64 = 1, mem_type := Memory.Default) -> rawptr
{
    vma_usage: vma.Memory_Usage
    properties: vk.MemoryPropertyFlags
    switch mem_type
    {
        case .Default:
        {
            properties = { .HOST_VISIBLE, .HOST_COHERENT }
            vma_usage = .Cpu_To_Gpu
        }
        case .GPU:
        {
            properties = { .DEVICE_LOCAL }
            vma_usage = .Gpu_Only
        }
        case .Readback:
        {
            properties = { .HOST_VISIBLE, .HOST_CACHED, .HOST_COHERENT }
            vma_usage = .Gpu_To_Cpu
        }
    }

    buf_usage: vk.BufferUsageFlags
    if mem_type == .GPU {
        buf_usage = { .SHADER_DEVICE_ADDRESS, .INDEX_BUFFER, .STORAGE_BUFFER, .TRANSFER_DST, .INDIRECT_BUFFER, .RESOURCE_DESCRIPTOR_BUFFER_EXT }
    } else {
        buf_usage = { .RESOURCE_DESCRIPTOR_BUFFER_EXT, .SHADER_DEVICE_ADDRESS, .STORAGE_BUFFER, .TRANSFER_SRC, .INDIRECT_BUFFER }
    }
    buf_ci := vk.BufferCreateInfo {
        sType = .BUFFER_CREATE_INFO,
        size = cast(vk.DeviceSize) bytes,
        usage = buf_usage,
        sharingMode = .EXCLUSIVE,
    }

    buf: vk.Buffer
    vk_check(vk.CreateBuffer(ctx.device, &buf_ci, nil, &buf))

    mem_requirements: vk.MemoryRequirements
    vk.GetBufferMemoryRequirements(ctx.device, buf, &mem_requirements)

    mem_requirements.alignment = vk.DeviceSize(max(u64(mem_requirements.alignment), align))

    alloc_ci := vma.Allocation_Create_Info {
        flags = vma.Allocation_Create_Flags { .Mapped } if mem_type != .GPU else {},
        usage = vma_usage,
        required_flags = properties,
    }
    alloc: vma.Allocation
    alloc_info: vma.Allocation_Info
    vk_check(vma.allocate_memory(ctx.vma_allocator, mem_requirements, alloc_ci, &alloc, &alloc_info))

    vk_check(vma.bind_buffer_memory(ctx.vma_allocator, alloc, buf))

    info := vk.BufferDeviceAddressInfo {
        sType = .BUFFER_DEVICE_ADDRESS_INFO,
        buffer = buf
    }
    addr := vk.GetBufferDeviceAddress(ctx.device, &info)
    addr_ptr := cast(rawptr) cast(uintptr) addr

    append(&ctx.gpu_allocs, GPU_Alloc_Meta {
        allocation = alloc,
        buf_handle = buf,
        device_address = addr,
        align = u32(align),
        buf_size = cast(vk.DeviceSize) bytes,
    })
    gpu_alloc_idx := u32(len(ctx.gpu_allocs)) - 1
    ctx.gpu_ptr_to_alloc[addr_ptr] = gpu_alloc_idx
    rbt.find_or_insert(&ctx.alloc_tree, Alloc_Range { u64(addr), u32(bytes) }, gpu_alloc_idx)

    if mem_type != .GPU
    {
        ptr := alloc_info.mapped_data
        ctx.cpu_ptr_to_alloc[ptr] = gpu_alloc_idx
        return ptr
    }

    return rawptr(uintptr(addr))
}

_mem_free :: proc(ptr: rawptr, loc := #caller_location)
{
    cpu_alloc, cpu_found := ctx.cpu_ptr_to_alloc[ptr]
    gpu_alloc, gpu_found := ctx.gpu_ptr_to_alloc[ptr]
    if !cpu_found && !gpu_found
    {
        log.error("Attempting to free a pointer which is not allocated.", location = loc)
        return
    }

    if cpu_found
    {
        meta := ctx.gpu_allocs[cpu_alloc]
        vma.destroy_buffer(ctx.vma_allocator, meta.buf_handle, meta.allocation)
        delete_key(&ctx.cpu_ptr_to_alloc, ptr)
    }
    else if gpu_found
    {
        meta := ctx.gpu_allocs[gpu_alloc]
        vma.destroy_buffer(ctx.vma_allocator, meta.buf_handle, meta.allocation)
        delete_key(&ctx.gpu_ptr_to_alloc, ptr)
    }
}

_host_to_device_ptr :: proc(ptr: rawptr) -> rawptr
{
    // We could do a tree search here but that would be more expensive

    meta_idx, found := ctx.cpu_ptr_to_alloc[ptr]
    if !found
    {
        log.error("Attempting to get the device pointer of a host pointer which is not allocated. Note: The pointer passed to this function must be a base allocation pointer.")
        return {}
    }

    meta := ctx.gpu_allocs[meta_idx]
    return rawptr(uintptr(meta.device_address))
}

// Textures
_texture_create :: proc(desc: Texture_Desc, storage: rawptr, signal_sem: Semaphore = {}, signal_value: u64 = 0) -> Texture
{
    vk_signal_sem := transmute(vk.Semaphore) signal_sem

    alloc_idx, ok_s := search_alloc_from_gpu_ptr(storage)
    if !ok_s
    {
        log.error("Address does not reside in allocated GPU memory.")
        return {}
    }
    alloc := ctx.gpu_allocs[alloc_idx]

    image: vk.Image
    offset := uintptr(storage) - uintptr(alloc.device_address)
    vk_check(vma.create_aliasing_image2(ctx.vma_allocator, alloc.allocation, vk.DeviceSize(offset), {
        sType = .IMAGE_CREATE_INFO,
        imageType = to_vk_texture_type(desc.type),
        format = to_vk_texture_format(desc.format),
        extent = vk.Extent3D { desc.dimensions.x, desc.dimensions.y, desc.dimensions.z },
        mipLevels = desc.mip_count,
        arrayLayers = desc.layer_count,
        samples = to_vk_sample_count(desc.sample_count),
        usage = to_vk_texture_usage(desc.usage) + { .TRANSFER_DST },
        initialLayout = .UNDEFINED,
    }, &image))

    plane_aspect: vk.ImageAspectFlags = { .DEPTH } if desc.format == .D32_Float else { .COLOR }

    // Transition layout from UNDEFINED to GENERAL
    {
        cmd_buf := vk_acquire_cmd_buf(ctx.queue)

        cmd_buf_bi := vk.CommandBufferBeginInfo {
            sType = .COMMAND_BUFFER_BEGIN_INFO,
            flags = { .ONE_TIME_SUBMIT },
        }
        vk_check(vk.BeginCommandBuffer(cmd_buf, &cmd_buf_bi))

        transition := vk.ImageMemoryBarrier2 {
            sType = .IMAGE_MEMORY_BARRIER_2,
            image = image,
            subresourceRange = {
                aspectMask = plane_aspect,
                levelCount = 1,
                layerCount = 1,
            },
            oldLayout = .UNDEFINED,
            newLayout = .GENERAL,
            srcStageMask = { .ALL_COMMANDS },
            srcAccessMask = { .MEMORY_WRITE },
            dstStageMask = { .ALL_COMMANDS },
            dstAccessMask = { .MEMORY_READ, .MEMORY_WRITE },
        }
        vk.CmdPipelineBarrier2(cmd_buf, &vk.DependencyInfo {
            sType = .DEPENDENCY_INFO,
            imageMemoryBarrierCount = 1,
            pImageMemoryBarriers = &transition,
        })

        vk_check(vk.EndCommandBuffer(cmd_buf))
        vk_submit_cmd_buf(ctx.queue, cmd_buf, vk_signal_sem, signal_value)
    }

    return {
        dimensions = desc.dimensions,
        format = desc.format,
        handle = transmute(Texture_Handle) image
    }
}

_texture_destroy :: proc(texture: ^Texture)
{
    vk_image := transmute(vk.Image) texture.handle

    views := &ctx.image_views[vk_image]
    for view in views {
        vk.DestroyImageView(ctx.device, view.view, nil)
    }

    vk.DestroyImage(ctx.device, vk_image, nil)
    texture^ = {}
}

_texture_size_and_align :: proc(desc: Texture_Desc) -> (size: u64, align: u64)
{
    image_ci := vk.ImageCreateInfo {
        sType = .IMAGE_CREATE_INFO,
        imageType = to_vk_texture_type(desc.type),
        format = to_vk_texture_format(desc.format),
        extent = vk.Extent3D { desc.dimensions.x, desc.dimensions.y, desc.dimensions.z },
        mipLevels = desc.mip_count,
        arrayLayers = desc.layer_count,
        samples = to_vk_sample_count(desc.sample_count),
        usage = to_vk_texture_usage(desc.usage),
        initialLayout = .UNDEFINED,
    }

    plane_aspect: vk.ImageAspectFlags = { .DEPTH } if desc.format == .D32_Float else { .COLOR }

    info := vk.DeviceImageMemoryRequirements {
        sType = .DEVICE_IMAGE_MEMORY_REQUIREMENTS,
        pCreateInfo = &image_ci,
        planeAspect = plane_aspect,
    }

    mem_requirements_2 := vk.MemoryRequirements2 { sType = .MEMORY_REQUIREMENTS_2 }
    vk.GetDeviceImageMemoryRequirements(ctx.device, &info, &mem_requirements_2)

    mem_requirements := mem_requirements_2.memoryRequirements
    return u64(mem_requirements.size), u64(mem_requirements.alignment)
}

@(private="file")
get_or_add_image_view :: proc(image: vk.Image, info: vk.ImageViewCreateInfo) -> vk.ImageView
{
    entry, found := &ctx.image_views[image]
    if !found
    {
        ctx.image_views[image] = {}
        image_view: vk.ImageView
        view_ci := info
        vk_check(vk.CreateImageView(ctx.device, &view_ci, nil, &image_view))
        append(&ctx.image_views[image], Image_View_Info { info, image_view })
        return image_view
    }
    else
    {
        for view in entry
        {
            if view.info == info {
                return view.view
            }
        }

        image_view: vk.ImageView
        view_ci := info
        vk_check(vk.CreateImageView(ctx.device, &view_ci, nil, &image_view))
        append(entry, Image_View_Info { info, image_view })
        return image_view
    }
}

_texture_view_descriptor :: proc(texture: Texture, view_desc: Texture_View_Desc) -> Texture_Descriptor
{
    vk_image := transmute(vk.Image) texture.handle

    format := view_desc.format
    if format == .Default {
        format = texture.format
    }

    plane_aspect: vk.ImageAspectFlags = { .DEPTH } if format == .D32_Float else { .COLOR }

    image_view_ci := vk.ImageViewCreateInfo {
        sType = .IMAGE_VIEW_CREATE_INFO,
        image = vk_image,
        viewType = to_vk_texture_view_type(view_desc.type),
        format = to_vk_texture_format(format),
        subresourceRange = {
            aspectMask = plane_aspect,
            levelCount = 1,
            layerCount = 1,
        }
    }
    view := get_or_add_image_view(vk_image, image_view_ci)

    desc: Texture_Descriptor
    info := vk.DescriptorGetInfoEXT {
        sType = .DESCRIPTOR_GET_INFO_EXT,
        type = .SAMPLED_IMAGE,
        data = { pSampledImage = &{ sampler = {}, imageView = view, imageLayout = .GENERAL } }
    }
    vk.GetDescriptorEXT(ctx.device, &info, int(ctx.texture_desc_size), &desc)
    return desc
}

//_texture_rw_view_descriptor :: proc(texture: Texture, view_desc: Texture_View_Desc) -> Texture_Descriptor { return {} }

_sampler_descriptor :: proc(sampler_desc: Sampler_Desc) -> Sampler_Descriptor
{
    sampler_ci := vk.SamplerCreateInfo {
        sType = .SAMPLER_CREATE_INFO,
        magFilter = to_vk_filter(sampler_desc.mag_filter),
        minFilter = to_vk_filter(sampler_desc.min_filter),
        mipmapMode = to_vk_mipmap_filter(sampler_desc.mip_filter),
        addressModeU = to_vk_address_mode(sampler_desc.address_mode_u),
        addressModeV = to_vk_address_mode(sampler_desc.address_mode_v),
        addressModeW = to_vk_address_mode(sampler_desc.address_mode_w),
    }
    sampler := get_or_add_sampler(sampler_ci)

    desc: Sampler_Descriptor
    info := vk.DescriptorGetInfoEXT {
        sType = .DESCRIPTOR_GET_INFO_EXT,
        type = .SAMPLER,
        data = { pSampledImage = &{ sampler = sampler, imageView = {}, imageLayout = .GENERAL } }
    }
    vk.GetDescriptorEXT(ctx.device, &info, int(ctx.sampler_desc_size), &desc)
    return desc

    get_or_add_sampler :: proc(info: vk.SamplerCreateInfo) -> vk.Sampler
    {
        for sampler in ctx.samplers
        {
            if sampler.info == info {
                return sampler.sampler
            }
        }

        sampler: vk.Sampler
        sampler_ci := info
        vk_check(vk.CreateSampler(ctx.device, &sampler_ci, nil, &sampler))
        append(&ctx.samplers, Sampler_Info { info, sampler })
        return sampler
    }
}

_get_texture_view_descriptor_size :: proc() -> u32
{
    return ctx.texture_desc_size
}

_get_texture_rw_view_descriptor_size :: proc() -> u32
{
    return ctx.texture_rw_desc_size
}

_get_sampler_descriptor_size :: proc() -> u32
{
    return ctx.sampler_desc_size
}

// Shaders
_shader_create :: proc(code: []u32, type: Shader_Type) -> Shader
{
    vk_stage := to_vk_shader_stage(type)

    push_constant_ranges := []vk.PushConstantRange {
        {
            stageFlags = { .VERTEX, .FRAGMENT },
            size = Push_Constant_Size,
        }
    }

    desc_layouts := []vk.DescriptorSetLayout {
        ctx.textures_desc_layout,
        ctx.textures_rw_desc_layout,
        ctx.samplers_desc_layout,
        ctx.data_desc_layout,
        ctx.indirect_data_desc_layout,
    }

    shader_cis := vk.ShaderCreateInfoEXT {
        sType = .SHADER_CREATE_INFO_EXT,
        codeType = .SPIRV,
        codeSize = len(code) * size_of(code[0]),
        pCode = raw_data(code),
        pName = "main",
        stage = vk_stage,
        nextStage = vk.ShaderStageFlags { .FRAGMENT } if type == .Vertex else {},
        pushConstantRangeCount = u32(len(push_constant_ranges)),
        pPushConstantRanges = raw_data(push_constant_ranges),
        setLayoutCount = u32(len(desc_layouts)),
        pSetLayouts = raw_data(desc_layouts),
    }
    shader: vk.ShaderEXT
    vk_check(vk.CreateShadersEXT(ctx.device, 1, &shader_cis, nil, &shader))
    return transmute(Shader) shader
}

_shader_destroy :: proc(shader: ^Shader)
{
    vk_shader := transmute(vk.ShaderEXT) (shader^)
    vk.DestroyShaderEXT(ctx.device, vk_shader, nil)
    shader^ = {}
}

// Semaphores
_semaphore_create :: proc(init_value: u64 = 0) -> Semaphore
{
    next: rawptr
    next = &vk.SemaphoreTypeCreateInfo {
        sType = .SEMAPHORE_TYPE_CREATE_INFO,
        pNext = next,
        semaphoreType = .TIMELINE,
        initialValue = init_value,
    }
    sem_ci := vk.SemaphoreCreateInfo {
        sType = .SEMAPHORE_CREATE_INFO,
        pNext = next
    }
    sem: vk.Semaphore
    vk_check(vk.CreateSemaphore(ctx.device, &sem_ci, nil, &sem))

    return cast(Semaphore) uintptr(sem)
}

_semaphore_wait :: proc(sem: Semaphore, wait_value: u64)
{
    sems := []vk.Semaphore { auto_cast uintptr(sem) }
    values := []u64 { wait_value }
    assert(len(sems) == len(values))
    vk.WaitSemaphores(ctx.device, &{
        sType = .SEMAPHORE_WAIT_INFO,
        semaphoreCount = u32(len(sems)),
        pSemaphores = raw_data(sems),
        pValues = raw_data(values),
    }, timeout = max(u64))
}

_semaphore_destroy :: proc(sem: ^Semaphore)
{
    vk_sem := transmute(vk.Semaphore) (sem^)
    vk.DestroySemaphore(ctx.device, vk_sem, nil)
    sem^ = {}
}

// Command buffer

_get_queue :: proc() -> Queue
{
    return cast(Queue) ctx.queue
}

// Vulkan handle getters for external use (e.g., imgui integration)
_get_vulkan_instance :: proc() -> vk.Instance
{
    return ctx.instance
}

_get_vulkan_physical_device :: proc() -> vk.PhysicalDevice
{
    return ctx.phys_device
}

_get_vulkan_device :: proc() -> vk.Device
{
    return ctx.device
}

_get_vulkan_queue :: proc() -> vk.Queue
{
    return ctx.queue
}

_get_vulkan_queue_family :: proc() -> u32
{
    return ctx.queue_family_idx
}

_get_vulkan_command_buffer :: proc(cmd_buf: Command_Buffer) -> vk.CommandBuffer
{
    return cast(vk.CommandBuffer) cmd_buf
}

_get_swapchain_image_count :: proc() -> u32
{
    return u32(len(ctx.swapchain.images))
}

_commands_begin :: proc(queue: Queue) -> Command_Buffer
{
    vk_queue := cast(vk.Queue) queue

    cmd_buf := vk_acquire_cmd_buf(vk_queue)

    cmd_buf_bi := vk.CommandBufferBeginInfo {
        sType = .COMMAND_BUFFER_BEGIN_INFO,
        flags = { .ONE_TIME_SUBMIT },
    }
    vk_check(vk.BeginCommandBuffer(cmd_buf, &cmd_buf_bi))

    return cast(Command_Buffer) cmd_buf
}

_queue_submit :: proc(queue: Queue, cmd_bufs: []Command_Buffer, signal_sem: Semaphore = {}, signal_value: u64 = 0)
{
    vk_queue := cast(vk.Queue) queue
    vk_signal_sem := transmute(vk.Semaphore) signal_sem

    for cmd_buf in cmd_bufs
    {
        vk_cmd_buf := cast(vk.CommandBuffer) cmd_buf
        vk_check(vk.EndCommandBuffer(vk_cmd_buf))

        vk_submit_cmd_buf(vk_queue, vk_cmd_buf, vk_signal_sem, signal_value)
    }
}

// Commands

_cmd_mem_copy :: proc(cmd_buf: Command_Buffer, src, dst: rawptr, #any_int bytes: i64)
{
    vk_cmd_buf := cast(vk.CommandBuffer) cmd_buf

    src_buf, src_offset, ok_s := compute_buf_offset_from_gpu_ptr(src)
    dst_buf, dst_offset, ok_d := compute_buf_offset_from_gpu_ptr(dst)
    if !ok_s || !ok_d
    {
        log.error("Alloc not found.")
        return
    }

    copy_regions := []vk.BufferCopy {
        {
            srcOffset = vk.DeviceSize(src_offset),
            dstOffset = vk.DeviceSize(dst_offset),
            size = vk.DeviceSize(bytes),
        }
    }
    vk.CmdCopyBuffer(vk_cmd_buf, src_buf, dst_buf, u32(len(copy_regions)), raw_data(copy_regions))
}

// TODO: dst is ignored atm.
_cmd_copy_to_texture :: proc(cmd_buf: Command_Buffer, texture: Texture, src, dst: rawptr)
{
    vk_cmd_buf := cast(vk.CommandBuffer) cmd_buf
    vk_image := transmute(vk.Image) texture.handle

    src_buf, src_offset, ok_s := compute_buf_offset_from_gpu_ptr(src)
    if !ok_s {
        log.error("Alloc not found.")
        return
    }

    plane_aspect: vk.ImageAspectFlags = { .DEPTH } if texture.format == .D32_Float else { .COLOR }

    vk.CmdCopyBufferToImage(vk_cmd_buf, src_buf, vk_image, .GENERAL, 1, &vk.BufferImageCopy {
        bufferOffset = vk.DeviceSize(src_offset),
        bufferRowLength = texture.dimensions.x,
        bufferImageHeight = texture.dimensions.y,
        imageSubresource = {
            aspectMask = plane_aspect,
            mipLevel = 0,
            baseArrayLayer = 0,
            layerCount = 1,
        },
        imageOffset = {},
        imageExtent = { texture.dimensions.x, texture.dimensions.y, texture.dimensions.z }
    })
}

_cmd_set_texture_heap :: proc(cmd_buf: Command_Buffer, textures, textures_rw, samplers: rawptr)
{
    vk_cmd_buf := cast(vk.CommandBuffer) cmd_buf

    if textures == nil && textures_rw == nil && samplers == nil do return

    infos: [3]vk.DescriptorBufferBindingInfoEXT
    // Fill in infos with the subset of valid pointers
    cursor := u32(0)
    if textures != nil
    {
        infos[cursor] = {
            sType = .DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
            address = transmute(vk.DeviceAddress) textures,
            usage = { .RESOURCE_DESCRIPTOR_BUFFER_EXT, .SHADER_DEVICE_ADDRESS, .STORAGE_BUFFER, .TRANSFER_SRC },
        }
        cursor += 1
    }
    if textures_rw != nil
    {
        infos[cursor] = {
            sType = .DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
            address = transmute(vk.DeviceAddress) textures_rw,
            usage = { .RESOURCE_DESCRIPTOR_BUFFER_EXT, .SHADER_DEVICE_ADDRESS, .STORAGE_BUFFER, .TRANSFER_SRC },
        }
        cursor += 1
    }
    if samplers != nil
    {
        infos[cursor] = {
            sType = .DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
            address = transmute(vk.DeviceAddress) samplers,
            usage = { .RESOURCE_DESCRIPTOR_BUFFER_EXT, .SHADER_DEVICE_ADDRESS, .STORAGE_BUFFER, .TRANSFER_SRC },
        }
        cursor += 1
    }

    vk.CmdBindDescriptorBuffersEXT(vk_cmd_buf, cursor, &infos[0])

    buffer_offsets := []vk.DeviceSize { 0, 0, 0 }
    cursor = 0
    if textures != nil {
        vk.CmdSetDescriptorBufferOffsetsEXT(vk_cmd_buf, .GRAPHICS, ctx.common_pipeline_layout, 0, 1, &cursor, &buffer_offsets[0])
        cursor += 1
    }
    if textures_rw != nil {
        vk.CmdSetDescriptorBufferOffsetsEXT(vk_cmd_buf, .GRAPHICS, ctx.common_pipeline_layout, 1, 1, &cursor, &buffer_offsets[1])
        cursor += 1
    }
    if samplers != nil {
        vk.CmdSetDescriptorBufferOffsetsEXT(vk_cmd_buf, .GRAPHICS, ctx.common_pipeline_layout, 2, 1, &cursor, &buffer_offsets[2])
        cursor += 1
    }
}

_cmd_barrier :: proc(cmd_buf: Command_Buffer, before: Stage, after: Stage, hazards: Hazard_Flags = {})
{
    vk_cmd_buf := cast(vk.CommandBuffer) cmd_buf

    vk_before := to_vk_stage(before)
    vk_after  := to_vk_stage(after)

    barrier := vk.MemoryBarrier {
        sType = .MEMORY_BARRIER,
        srcAccessMask = { .MEMORY_WRITE },
        dstAccessMask = { .MEMORY_READ }
    }
    vk.CmdPipelineBarrier(vk_cmd_buf, vk_before, vk_after, {}, 1, &barrier, 0, nil, 0, nil)
}

_cmd_signal_after :: proc() {}
_cmd_wait_before :: proc() {}

_cmd_set_shaders :: proc(cmd_buf: Command_Buffer, vert_shader: Shader, frag_shader: Shader)
{
    vk_cmd_buf := cast(vk.CommandBuffer) cmd_buf
    vk_vert_shader := transmute(vk.ShaderEXT) vert_shader
    vk_frag_shader := transmute(vk.ShaderEXT) frag_shader

    shader_stages := []vk.ShaderStageFlags { { .VERTEX }, { .FRAGMENT } }
    to_bind := []vk.ShaderEXT { vk_vert_shader, vk_frag_shader }
    assert(len(shader_stages) == len(to_bind))
    vk.CmdBindShadersEXT(vk_cmd_buf, u32(len(shader_stages)), raw_data(shader_stages), raw_data(to_bind))
}

_cmd_set_depth_state :: proc(cmd_buf: Command_Buffer, state: Depth_State)
{
    vk_cmd_buf := cast(vk.CommandBuffer) cmd_buf

    vk.CmdSetDepthCompareOp(vk_cmd_buf, to_vk_compare_op(state.compare))
    vk.CmdSetDepthTestEnable(vk_cmd_buf, .Read in state.mode)
    vk.CmdSetDepthWriteEnable(vk_cmd_buf, .Write in state.mode)
    vk.CmdSetDepthBiasEnable(vk_cmd_buf, false)
    vk.CmdSetDepthClipEnableEXT(vk_cmd_buf, true)
    vk.CmdSetStencilTestEnable(vk_cmd_buf, false)
}

_cmd_set_blend_state :: proc(cmd_buf: Command_Buffer, state: Blend_State)
{
    vk_cmd_buf := cast(vk.CommandBuffer) cmd_buf

    enable_b32 := b32(state.enable)
    vk.CmdSetColorBlendEnableEXT(vk_cmd_buf, 0, 1, &enable_b32)

    vk.CmdSetColorBlendEquationEXT(vk_cmd_buf, 0, 1, &vk.ColorBlendEquationEXT {
        srcColorBlendFactor = {},
        dstColorBlendFactor = {},
        colorBlendOp        = {},
        srcAlphaBlendFactor = {},
        dstAlphaBlendFactor = {},
        alphaBlendOp        = {},
    })

    color_write_mask := transmute(vk.ColorComponentFlags) cast(u32) state.color_write_mask
    vk.CmdSetColorWriteMaskEXT(vk_cmd_buf, 0, 1, &color_write_mask)
}

// _cmd_dispatch :: proc() {}
// _cmd_dispatch_indirect :: proc() {}

_cmd_begin_render_pass :: proc(cmd_buf: Command_Buffer, desc: Render_Pass_Desc)
{
    vk_cmd_buf := cast(vk.CommandBuffer) cmd_buf

    scratch, _ := acquire_scratch()

    vk_color_attachments := make([]vk.RenderingAttachmentInfo, len(desc.color_attachments), allocator = scratch)
    for &vk_attach, i in vk_color_attachments {
        vk_attach = to_vk_render_attachment(desc.color_attachments[i])
    }

    vk_depth_attachment: vk.RenderingAttachmentInfo
    vk_depth_attachment_ptr: ^vk.RenderingAttachmentInfo
    if desc.depth_attachment != nil
    {
        vk_depth_attachment = to_vk_render_attachment(desc.depth_attachment.?)
        vk_depth_attachment_ptr = &vk_depth_attachment
    }

    width := desc.render_area_size.x
    if width == {} {
        width = desc.color_attachments[0].texture.dimensions.x
    }
    height := desc.render_area_size.y
    if height == {} {
        height = desc.color_attachments[0].texture.dimensions.y
    }
    layer_count := desc.layer_count
    if layer_count == 0 {
        layer_count = 1
    }

    rendering_info := vk.RenderingInfo {
        sType = .RENDERING_INFO,
        renderArea = {
            offset = { desc.render_area_offset.x, desc.render_area_offset.y },
            extent = { width, height }
        },
        layerCount = layer_count,
        colorAttachmentCount = u32(len(vk_color_attachments)),
        pColorAttachments = raw_data(vk_color_attachments),
        pDepthAttachment = vk_depth_attachment_ptr,
    }
    vk.CmdBeginRendering(vk_cmd_buf, &rendering_info)

    // Blend state
    vk.CmdSetStencilTestEnable(vk_cmd_buf, false)
    b32_false := b32(false)
    vk.CmdSetColorBlendEnableEXT(vk_cmd_buf, 0, 1, &b32_false)
    color_mask := vk.ColorComponentFlags { .R, .G, .B, .A }
    vk.CmdSetColorWriteMaskEXT(vk_cmd_buf, 0, 1, &color_mask)

    // Depth state
    vk.CmdSetDepthCompareOp(vk_cmd_buf, .LESS)
    vk.CmdSetDepthTestEnable(vk_cmd_buf, false)
    vk.CmdSetDepthWriteEnable(vk_cmd_buf, false)
    vk.CmdSetDepthBiasEnable(vk_cmd_buf, false)
    vk.CmdSetDepthClipEnableEXT(vk_cmd_buf, true)

    // Viewport
    viewport := vk.Viewport {
        x = 0, y = 0,
        width = f32(width), height = f32(height),
        minDepth = 0.0, maxDepth = 1.0,
    }
    vk.CmdSetViewportWithCount(vk_cmd_buf, 1, &viewport)
    scissor := vk.Rect2D {
        offset = {
            x = 0, y = 0
        },
        extent = {
            width = width, height = height,
        }
    }
    vk.CmdSetScissorWithCount(vk_cmd_buf, 1, &scissor)
    vk.CmdSetRasterizerDiscardEnable(vk_cmd_buf, false)

    // Unused
    vk.CmdSetVertexInputEXT(vk_cmd_buf, 0, nil, 0, nil)
    vk.CmdSetRasterizationSamplesEXT(vk_cmd_buf, { ._1 })
    vk.CmdSetPrimitiveTopology(vk_cmd_buf, .TRIANGLE_LIST)
    vk.CmdSetPrimitiveRestartEnable(vk_cmd_buf, false)

    sample_mask := vk.SampleMask(1)
    vk.CmdSetSampleMaskEXT(vk_cmd_buf, { ._1 }, &sample_mask)
    vk.CmdSetAlphaToCoverageEnableEXT(vk_cmd_buf, false)
    vk.CmdSetPolygonModeEXT(vk_cmd_buf, .FILL)
    vk.CmdSetCullMode(vk_cmd_buf, { .BACK })
    vk.CmdSetFrontFace(vk_cmd_buf, .COUNTER_CLOCKWISE)
}

_cmd_end_render_pass :: proc(cmd_buf: Command_Buffer)
{
    vk_cmd_buf := cast(vk.CommandBuffer) cmd_buf
    vk.CmdEndRendering(vk_cmd_buf)
}

_cmd_draw_indexed_instanced :: proc(cmd_buf: Command_Buffer, vertex_data: rawptr, fragment_data: rawptr,
                                    indices: rawptr, index_count: u32, instance_count: u32 = 1)
{
    vk_cmd_buf := cast(vk.CommandBuffer) cmd_buf

    indices_buf, indices_offset, ok_i := compute_buf_offset_from_gpu_ptr(indices)
    if !ok_i
    {
        log.error("Indices alloc not found")
        return
    }

    // Push constants: vert_data, frag_data, vert_indirect_data, frag_indirect_data
    ptrs := []rawptr { vertex_data, fragment_data, vertex_data, fragment_data }
    assert(Push_Constant_Size == len(ptrs) * size_of(ptrs[0]))
    vk.CmdPushConstants(vk_cmd_buf, ctx.common_pipeline_layout, { .VERTEX, .FRAGMENT }, 0, Push_Constant_Size, raw_data(ptrs))

    vk.CmdBindIndexBuffer(vk_cmd_buf, indices_buf, vk.DeviceSize(indices_offset), .UINT32)
    vk.CmdDrawIndexed(vk_cmd_buf, index_count, instance_count, 0, 0, 0)
}

_cmd_draw_indexed_instanced_indirect :: proc(cmd_buf: Command_Buffer, vertex_data: rawptr, fragment_data: rawptr,
                                            indices: rawptr, arguments: rawptr)
{
    vk_cmd_buf := cast(vk.CommandBuffer) cmd_buf

    indices_buf, indices_offset, ok_i := compute_buf_offset_from_gpu_ptr(indices)
    if !ok_i
    {
        log.error("Indices alloc not found")
        return
    }

    arguments_buf, arguments_offset, ok_a := compute_buf_offset_from_gpu_ptr(arguments)
    if !ok_a
    {
        log.error("Arguments alloc not found")
        return
    }

    // Push constants: vert_data, frag_data, vert_indirect_data, frag_indirect_data
    ptrs := []rawptr { vertex_data, fragment_data, vertex_data, fragment_data }
    assert(Push_Constant_Size == len(ptrs) * size_of(ptrs[0]))
    vk.CmdPushConstants(vk_cmd_buf, ctx.common_pipeline_layout, { .VERTEX, .FRAGMENT }, 0, Push_Constant_Size, raw_data(ptrs))

    vk.CmdBindIndexBuffer(vk_cmd_buf, indices_buf, vk.DeviceSize(indices_offset), .UINT32)
    vk.CmdDrawIndexedIndirect(vk_cmd_buf, arguments_buf, vk.DeviceSize(arguments_offset), 1, 0)
}

@(private="file")
_cmd_draw_indexed_instanced_indirect_multi_impl :: proc(cmd_buf: Command_Buffer, data_vertex: rawptr, data_pixel: rawptr,
                                                         indices: rawptr, arguments: rawptr, draw_count: rawptr,
                                                         data_vertex_shared: rawptr, data_pixel_shared: rawptr)
{
    vk_cmd_buf := cast(vk.CommandBuffer) cmd_buf

    indices_buf, indices_offset, ok_i := compute_buf_offset_from_gpu_ptr(indices)
    if !ok_i
    {
        log.error("Indices alloc not found")
        return
    }

    arguments_buf, arguments_offset, ok_a := compute_buf_offset_from_gpu_ptr(arguments)
    if !ok_a
    {
        log.error("Arguments alloc not found")
        return
    }

    draw_count_buf, draw_count_offset, ok_dc := compute_buf_offset_from_gpu_ptr(draw_count)
    if !ok_dc
    {
        log.error("Draw count alloc not found")
        return
    }

    // Push constants contain: vert_data, frag_data, vert_indirect_data, frag_indirect_data
    ptrs := []rawptr { data_vertex_shared, data_pixel_shared, data_vertex, data_pixel }
    assert(Push_Constant_Size == len(ptrs) * size_of(ptrs[0]))
    vk.CmdPushConstants(vk_cmd_buf, ctx.common_pipeline_layout, { .VERTEX, .FRAGMENT }, 0, Push_Constant_Size, raw_data(ptrs))

    vk.CmdBindIndexBuffer(vk_cmd_buf, indices_buf, vk.DeviceSize(indices_offset), .UINT32)
    stride := u32(size_of(vk.DrawIndexedIndirectCommand))

    max_draw_count: u32 = 0xFFFFFFFF
    buf_size, ok_size := get_buf_size_from_gpu_ptr(arguments)
    if ok_size && buf_size > vk.DeviceSize(arguments_offset)
    {
        available_size := buf_size - vk.DeviceSize(arguments_offset)
        max_draw_count = u32(available_size / vk.DeviceSize(stride))
    }

    vk.CmdDrawIndexedIndirectCount(vk_cmd_buf, arguments_buf, vk.DeviceSize(arguments_offset), draw_count_buf, vk.DeviceSize(draw_count_offset), max_draw_count, stride)
}

_cmd_draw_indexed_instanced_indirect_multi :: proc(cmd_buf: Command_Buffer, data_vertex: rawptr, data_pixel: rawptr,
                                                    indices: rawptr, arguments: rawptr, draw_count: rawptr)
{
    _cmd_draw_indexed_instanced_indirect_multi_impl(cmd_buf, data_vertex, data_pixel, indices, arguments, draw_count, data_vertex, data_pixel)
}

_cmd_draw_indexed_instanced_indirect_multi_data :: proc(cmd_buf: Command_Buffer, data_vertex: rawptr, data_pixel: rawptr,
                                                         indices: rawptr, arguments: rawptr, draw_count: rawptr, data_vertex_shared: rawptr,
                                                         data_pixel_shared: rawptr)
{
    _cmd_draw_indexed_instanced_indirect_multi_impl(cmd_buf, data_vertex, data_pixel, indices, arguments, draw_count, data_vertex_shared, data_pixel_shared)
}

@(private="file")
vk_check :: proc(result: vk.Result, location := #caller_location)
{
    if result != .SUCCESS {
        fatal_error("Vulkan failure: %v", result, location = location)
    }
}

@(private="file")
vk_debug_callback :: proc "system" (severity: vk.DebugUtilsMessageSeverityFlagsEXT,
                                    types: vk.DebugUtilsMessageTypeFlagsEXT,
                                    callback_data: ^vk.DebugUtilsMessengerCallbackDataEXT,
                                    user_data: rawptr) -> b32
{
    context = runtime.default_context()
    context.logger = vk_logger

    level: log.Level
    if .ERROR in severity        do level = .Error
    else if .WARNING in severity do level = .Warning
    else if .INFO in severity    do level = .Info
    else                         do level = .Debug
    log.log(level, callback_data.pMessage)

    return false
}

@(private="file")
fatal_error :: proc(fmt: string, args: ..any, location := #caller_location)
{
    when ODIN_DEBUG {
        log.fatalf(fmt, ..args, location = location)
        runtime.panic("")
    } else {
        log.panicf(fmt, ..args, location = location)
    }
}

@(private="file")
align_up :: proc(x, align: u64) -> (aligned: u64)
{
    assert(0 == (align & (align - 1)), "must align to a power of two")
    return (x + (align - 1)) &~ (align - 1)
}

// Scratch arenas

@(private="file")
scratch_arenas: [4]vmem.Arena

@(private="file")
init_scratch_arenas :: proc()
{
    for &scratch in scratch_arenas
    {
        error := vmem.arena_init_growing(&scratch)
        assert(error == nil)
    }
}

@(private="file")
@(deferred_out = release_scratch)
acquire_scratch :: proc(used_allocators: ..mem.Allocator) -> (mem.Allocator, vmem.Arena_Temp)
{
    available_arena: ^vmem.Arena
    if len(used_allocators) < 1
    {
        available_arena = &scratch_arenas[0]
    }
    else
    {
        for &scratch in scratch_arenas
        {
            for used_alloc in used_allocators
            {
                // NOTE: We assume that if the data points to the same exact address,
                // it's an arena allocator and it's the same arena
                if used_alloc.data != &scratch
                {
                    available_arena = &scratch
                    break
                }

                if available_arena != nil do break
            }
        }
    }

    assert(available_arena != nil, "Available scratch arena not found.")

    return vmem.arena_allocator(available_arena), vmem.arena_temp_begin(available_arena)
}

@(private="file")
release_scratch :: #force_inline proc(allocator: mem.Allocator, temp: vmem.Arena_Temp)
{
    vmem.arena_temp_end(temp)
}

@(private="file")
create_swapchain :: proc(width: u32, height: u32, frames_in_flight: u32) -> Swapchain
{
    scratch, _ := acquire_scratch()

    res: Swapchain

    surface_caps: vk.SurfaceCapabilitiesKHR
    vk_check(vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(ctx.phys_device, ctx.surface, &surface_caps))

    image_count := max(max(2, surface_caps.minImageCount), frames_in_flight)
    if surface_caps.maxImageCount != 0 do assert(image_count <= surface_caps.maxImageCount)

    surface_format_count: u32
    vk_check(vk.GetPhysicalDeviceSurfaceFormatsKHR(ctx.phys_device, ctx.surface, &surface_format_count, nil))
    surface_formats := make([]vk.SurfaceFormatKHR, surface_format_count, allocator = scratch)
    vk_check(vk.GetPhysicalDeviceSurfaceFormatsKHR(ctx.phys_device, ctx.surface, &surface_format_count, raw_data(surface_formats)))

    surface_format := surface_formats[0]
    for candidate in surface_formats
    {
        if candidate == { .B8G8R8A8_UNORM, .SRGB_NONLINEAR }
        {
            surface_format = candidate
            break
        }
    }

    present_mode_count: u32
    vk_check(vk.GetPhysicalDeviceSurfacePresentModesKHR(ctx.phys_device, ctx.surface, &present_mode_count, nil))
    present_modes := make([]vk.PresentModeKHR, present_mode_count, allocator = scratch)
    vk_check(vk.GetPhysicalDeviceSurfacePresentModesKHR(ctx.phys_device, ctx.surface, &present_mode_count, raw_data(present_modes)))

    present_mode := vk.PresentModeKHR.FIFO
    for candidate in present_modes {
        if candidate == .MAILBOX {
            present_mode = candidate
            break
        }
    }

    res.width = width
    res.height = height

    swapchain_ci := vk.SwapchainCreateInfoKHR {
        sType = .SWAPCHAIN_CREATE_INFO_KHR,
        surface = ctx.surface,
        minImageCount = image_count,
        imageFormat = surface_format.format,
        imageColorSpace = surface_format.colorSpace,
        imageExtent = { res.width, res.height },
        imageArrayLayers = 1,
        imageUsage = { .COLOR_ATTACHMENT },
        preTransform = surface_caps.currentTransform,
        compositeAlpha = { .OPAQUE },
        presentMode = present_mode,
        clipped = true,
    }
    vk_check(vk.CreateSwapchainKHR(ctx.device, &swapchain_ci, nil, &res.handle))

    vk_check(vk.GetSwapchainImagesKHR(ctx.device, res.handle, &image_count, nil))
    res.images = make([]vk.Image, image_count, context.allocator)
    vk_check(vk.GetSwapchainImagesKHR(ctx.device, res.handle, &image_count, raw_data(res.images)))

    res.image_views = make([]vk.ImageView, image_count, context.allocator)
    for image, i in res.images
    {
        image_view_ci := vk.ImageViewCreateInfo {
            sType = .IMAGE_VIEW_CREATE_INFO,
            image = image,
            viewType = .D2,
            format = surface_format.format,
            subresourceRange = {
                aspectMask = { .COLOR },
                levelCount = 1,
                layerCount = 1,
            },
        }
        vk_check(vk.CreateImageView(ctx.device, &image_view_ci, nil, &res.image_views[i]))
    }

    res.present_semaphores = make([]vk.Semaphore, image_count, context.allocator)

    semaphore_ci := vk.SemaphoreCreateInfo { sType = .SEMAPHORE_CREATE_INFO }
    for &semaphore in res.present_semaphores {
        vk_check(vk.CreateSemaphore(ctx.device, &semaphore_ci, nil, &semaphore))
    }

    return res
}

@(private="file")
destroy_swapchain :: proc(swapchain: Swapchain)
{
    for image in swapchain.images
    {
        views := &ctx.image_views[image]
        for view in views {
            vk.DestroyImageView(ctx.device, view.view, nil)
        }
    }

    delete(swapchain.images)
    for semaphore in swapchain.present_semaphores {
        vk.DestroySemaphore(ctx.device, semaphore, nil)
    }
    delete(swapchain.present_semaphores)
    for image_view in swapchain.image_views {
        vk.DestroyImageView(ctx.device, image_view, nil)
    }
    delete(swapchain.image_views)
    vk.DestroySwapchainKHR(ctx.device, swapchain.handle, nil)
}

@(private="file")
Swapchain :: struct
{
    handle: vk.SwapchainKHR,
    width, height: u32,
    images: []vk.Image,
    image_views: []vk.ImageView,
    present_semaphores: []vk.Semaphore,
}

// NOTE: This is slow but unfortunately needed for some things. Vulkan
// is still a "buffer object" centric API.
@(private="file")
search_alloc_from_gpu_ptr :: proc(ptr: rawptr) -> (res: u32, ok: bool)
{
    alloc_idx, found := rbt.find_value(&ctx.alloc_tree, Alloc_Range { u64(uintptr(ptr)), 0 })
    return alloc_idx, found
}

@(private="file")
compute_buf_offset_from_gpu_ptr :: proc(ptr: rawptr) -> (buf: vk.Buffer, offset: u32, ok: bool)
{
    alloc_idx, ok_s := search_alloc_from_gpu_ptr(ptr)
    if !ok_s do return {}, {}, false

    alloc := ctx.gpu_allocs[alloc_idx]

    buf = alloc.buf_handle
    offset = u32(uintptr(ptr) - uintptr(alloc.device_address))
    return buf, offset, true
}

@(private="file")
get_buf_size_from_gpu_ptr :: proc(ptr: rawptr) -> (size: vk.DeviceSize, ok: bool)
{
    alloc_idx, ok_s := search_alloc_from_gpu_ptr(ptr)
    if !ok_s do return 0, false

    // Get actual buffer size from metadata (not allocation size, which may be larger due to alignment)
    alloc := ctx.gpu_allocs[alloc_idx]
    return alloc.buf_size, true
}

// Command buffers
@(private="file")
vk_acquire_cmd_buf :: proc(queue: vk.Queue) -> vk.CommandBuffer
{
    // Poll semaphores
    found_free := -1
    for _, i in ctx.cmd_bufs
    {
        if ctx.cmd_bufs_timelines[i].recording do continue

        sem := ctx.cmd_bufs_timelines[i].sem
        des_val := ctx.cmd_bufs_timelines[i].val
        val: u64
        vk.GetSemaphoreCounterValue(ctx.device, sem, &val)
        if val >= des_val
        {
            found_free = i
            ctx.cmd_bufs_timelines[i].recording = true
            break
        }
    }

    assert(found_free != -1)  // TODO

    return ctx.cmd_bufs[found_free]
}

@(private="file")
vk_submit_cmd_buf :: proc(queue: vk.Queue, cmd_buf: vk.CommandBuffer, signal_sem: vk.Semaphore = {}, signal_value: u64 = 0)
{
    // Find command buffer in array
    found_idx := -1
    for buf, i in ctx.cmd_bufs
    {
        if buf == cmd_buf
        {
            found_idx = i
            break
        }
    }

    assert(found_idx != -1)

    ctx.cmd_bufs_timelines[found_idx].val += 1

    cmd_buf_sem := ctx.cmd_bufs_timelines[found_idx].sem
    cmd_buf_sem_value := ctx.cmd_bufs_timelines[found_idx].val

    signal_sems: []vk.Semaphore = { cmd_buf_sem, signal_sem } if signal_sem != {} else { cmd_buf_sem }
    signal_values: []u64 = { cmd_buf_sem_value, signal_value } if signal_sem != {} else { cmd_buf_sem_value }

    next: rawptr
    next = &vk.TimelineSemaphoreSubmitInfo {
        sType = .TIMELINE_SEMAPHORE_SUBMIT_INFO,
        pNext = next,
        signalSemaphoreValueCount = u32(len(signal_values)),
        pSignalSemaphoreValues = raw_data(signal_values)
    }
    to_submit := []vk.CommandBuffer { cmd_buf }
    submit_info := vk.SubmitInfo {
        sType = .SUBMIT_INFO,
        pNext = next,
        commandBufferCount = u32(len(to_submit)),
        pCommandBuffers = raw_data(to_submit),
        signalSemaphoreCount = u32(len(signal_sems)),
        pSignalSemaphores = raw_data(signal_sems)
    }
    vk_check(vk.QueueSubmit(queue, 1, &submit_info, {}))

    ctx.cmd_bufs_timelines[found_idx].recording = false
}

@(private="file")
vk_get_cmd_buf_timeline :: proc(queue: vk.Queue, cmd_buf: vk.CommandBuffer) -> ^Timeline
{
    // Find command buffer in array
    found_idx := -1
    for buf, i in ctx.cmd_bufs
    {
        if buf == cmd_buf
        {
            found_idx = i
            break
        }
    }

    assert(found_idx != -1)
    return &ctx.cmd_bufs_timelines[found_idx]
}

// Enum conversion

@(private="file")
to_vk_shader_stage :: #force_inline proc(type: Shader_Type) -> vk.ShaderStageFlags
{
    switch type
    {
        case .Vertex: return { .VERTEX }
        case .Fragment: return { .FRAGMENT }
    }
    return {}
}

@(private="file")
to_vk_stage :: #force_inline proc(stage: Stage) -> vk.PipelineStageFlags
{
    switch stage
    {
        case .Transfer: return { .TRANSFER }
        case .Compute: return { .COMPUTE_SHADER }
        case .Raster_Color_Out: return { .COLOR_ATTACHMENT_OUTPUT }
        case .Fragment_Shader: return { .FRAGMENT_SHADER }
        case .Vertex_Shader: return { .VERTEX_SHADER }
        case .All: return { .ALL_COMMANDS }
    }
    return {}
}

@(private="file")
to_vk_load_op :: #force_inline proc(load_op: Load_Op) -> vk.AttachmentLoadOp
{
    switch load_op
    {
        case .Clear: return .CLEAR
        case .Load: return .LOAD
        case .Dont_Care: return .DONT_CARE
    }
    return {}
}

@(private="file")
to_vk_store_op :: #force_inline proc(store_op: Store_Op) -> vk.AttachmentStoreOp
{
    switch store_op
    {
        case .Store: return .STORE
        case .Dont_Care: return .DONT_CARE
    }
    return {}
}

@(private="file")
to_vk_compare_op :: #force_inline proc(compare_op: Compare_Op) -> vk.CompareOp
{
    switch compare_op
    {
        case .Never: return .NEVER
        case .Less: return .LESS
        case .Equal: return .EQUAL
        case .Less_Equal: return .LESS_OR_EQUAL
        case .Greater: return .GREATER
        case .Not_Equal: return .NOT_EQUAL
        case .Greater_Equal: return .GREATER_OR_EQUAL
        case .Always: return .ALWAYS
    }
    return {}
}

@(private="file")
to_vk_render_attachment :: #force_inline proc(attach: Render_Attachment) -> vk.RenderingAttachmentInfo
{
    view_desc := attach.view
    texture := attach.texture
    vk_image := transmute(vk.Image) texture.handle

    format := view_desc.format
    if format == .Default {
        format = attach.texture.format
    }

    plane_aspect: vk.ImageAspectFlags = { .DEPTH } if format == .D32_Float else { .COLOR }

    image_view_ci := vk.ImageViewCreateInfo {
        sType = .IMAGE_VIEW_CREATE_INFO,
        image = vk_image,
        viewType = to_vk_texture_view_type(view_desc.type),
        format = to_vk_texture_format(format),
        subresourceRange = {
            aspectMask = plane_aspect,
            levelCount = 1,
            layerCount = 1,
        }
    }
    view := get_or_add_image_view(vk_image, image_view_ci)

    return {
        sType = .RENDERING_ATTACHMENT_INFO,
        imageView = view,
        imageLayout = .GENERAL,
        loadOp = to_vk_load_op(attach.load_op),
        storeOp = to_vk_store_op(attach.store_op),
        clearValue = { color = { float32 = attach.clear_color } }
    }
}

@(private="file")
to_vk_texture_type :: #force_inline proc(type: Texture_Type) -> vk.ImageType
{
    switch type
    {
        case .D2: return .D2
        case .D3: return .D3
        case .D1: return .D1
    }
    return {}
}

@(private="file")
to_vk_texture_view_type :: #force_inline proc(type: Texture_Type) -> vk.ImageViewType
{
    switch type
    {
        case .D2: return .D2
        case .D3: return .D3
        case .D1: return .D1
    }
    return {}
}

to_vk_texture_format :: proc(format: Texture_Format) -> vk.Format
{
    switch format
    {
        case .Default: panic("Implementation bug!")
        case .RGBA8_Unorm: return .R8G8B8A8_UNORM
        case .BGRA8_Unorm: return .B8G8R8A8_UNORM
        case .D32_Float: return .D32_SFLOAT
    }
    return {}
}

to_vk_sample_count :: proc(sample_count: u32) -> vk.SampleCountFlags
{
    switch sample_count
    {
        case 0: return { ._1 }
        case 1: return { ._1 }
        case 2: return { ._2 }
        case 4: return { ._4 }
        case 8: return { ._8 }
        case: panic("Unsupported sample count.")
    }
    return {}
}

to_vk_texture_usage :: proc(usage: Usage_Flags) -> vk.ImageUsageFlags
{
    res: vk.ImageUsageFlags
    if .Sampled in usage do                  res += { .SAMPLED }
    if .Storage in usage do                  res += { .STORAGE }
    if .Color_Attachment in usage do         res += { .COLOR_ATTACHMENT }
    if .Depth_Stencil_Attachment in usage do res += { .DEPTH_STENCIL_ATTACHMENT }
    return res
}

to_vk_filter :: proc(filter: Filter) -> vk.Filter
{
    switch filter
    {
        case .Linear: return .LINEAR
        case .Nearest: return .NEAREST
    }
    return {}
}

to_vk_mipmap_filter :: proc(filter: Filter) -> vk.SamplerMipmapMode
{
    switch filter
    {
        case .Linear: return .LINEAR
        case .Nearest: return .NEAREST
    }
    return {}
}

to_vk_address_mode :: proc(addr_mode: Address_Mode) -> vk.SamplerAddressMode
{
    switch addr_mode
    {
        case .Repeat: return .REPEAT
        case .Mirrored_Repeat: return .MIRRORED_REPEAT
        case .Clamp_To_Edge: return .CLAMP_TO_EDGE
    }
    return {}
}

@(private="file")
Pool_Element :: struct($T: typeid)
{
    using el: T,
    present: bool,
}

@(private="file")
Pool :: struct($T: typeid)
{
    array: [dynamic]Pool_Element(T),
    free_list: [dynamic]u32,
}

@(private="file")
pool_append :: proc(using pool: ^Pool($T), el: T) -> u32
{
    free_idx: u32
    if len(free_list) > 0
    {
        free_idx = pop(&free_list)
    }
    else
    {
        append(&array, {})
        free_idx = len(array) - 1
    }

    array[free_idx].el = el
    array[free_idx].present = true
    return free_idx
}

@(private="file")
pool_free_idx :: proc(using pool: ^Pool($T), idx: u32)
{
    if idx == len(array)
    {
        pop(&array)
    }
    else
    {
        array[idx].present = false
        append(&free_list, idx)
    }
}

@(private="file")
pool_destroy :: proc(using pool: ^Pool($T))
{
    delete(array)
    delete(free_list)
    array = {}
    free_list = {}
}
