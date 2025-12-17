
package gpu

import "core:slice"

import sdl "vendor:sdl3"
import vk "vendor:vulkan"

Context :: struct
{
    instance: vk.Instance,
    debug_messenger: vk.DebugUtilsMessengerEXT,
    surface: vk.SurfaceKHR,

    phys_device: vk.PhysicalDevice,
    device: vk.Device,
    queue: vk.Queue,
    lm_queue: vk.Queue,
    queue_family_idx: u32,
}

ctx: Context

init :: proc(window: ^sdl.Window)
{
    res: Context

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
                vk.KHR_WIN32_SURFACE_EXTENSION_NAME,
            }
        }, context.temp_allocator)

        debug_messenger_ci := vk.DebugUtilsMessengerCreateInfoEXT {
            sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            messageSeverity = { .WARNING, .ERROR },
            messageType = { .VALIDATION, .PERFORMANCE },
            pfnUserCallback = debug_callback
        }

        when ODIN_DEBUG
        {
            validation_features := []vk.ValidationFeatureEnableEXT {
                //.GPU_ASSISTED,
                //.GPU_ASSISTED_RESERVE_BINDING_SLOT,
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
        }, nil, &res.instance))

        vk.load_proc_addresses_instance(res.instance)
        assert(vk.DestroyInstance != nil, "Failed to load Vulkan instance API")

        vk_check(vk.CreateDebugUtilsMessengerEXT(res.instance, &debug_messenger_ci, nil, &res.debug_messenger))
    }

    // Create surface
    {
        ok_s := sdl.Vulkan_CreateSurface(window, res.instance, nil, &res.surface)
        if !ok_s do fatal_error("Could not create vulkan surface.")
    }

    // Physical device
    phys_device_count: u32
    vk_check(vk.EnumeratePhysicalDevices(res.instance, &phys_device_count, nil))
    if phys_device_count == 0 do fatal_error("Did not find any GPUs!")
    phys_devices := make([]vk.PhysicalDevice, phys_device_count, context.temp_allocator)
    vk_check(vk.EnumeratePhysicalDevices(res.instance, &phys_device_count, raw_data(phys_devices)))

    chosen_phys_device: vk.PhysicalDevice
    queue_family_idx: u32
    found := false
    device_loop: for candidate in phys_devices
    {
        queue_family_count: u32
        vk.GetPhysicalDeviceQueueFamilyProperties(candidate, &queue_family_count, nil)
        queue_families := make([]vk.QueueFamilyProperties, queue_family_count, context.temp_allocator)
        vk.GetPhysicalDeviceQueueFamilyProperties(candidate, &queue_family_count, raw_data(queue_families))

        for family, i in queue_families
        {
            supports_graphics := .GRAPHICS in family.queueFlags
            supports_present: b32
            vk_check(vk.GetPhysicalDeviceSurfaceSupportKHR(candidate, u32(i), res.surface, &supports_present))

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

    res.phys_device = chosen_phys_device

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
        vk.KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        vk.KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        vk.KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        vk.EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME,
        vk.KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME,
        vk.KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
    }

    next: rawptr
    /*
    next = &vk.PhysicalDeviceMaintenance6Features {
        sType = .PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES,
        pNext = next,
        maintenance6 = true,
    }
    */
    next = &vk.PhysicalDeviceVulkan12Features {
        sType = .PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        pNext = next,
        runtimeDescriptorArray = true,
        shaderSampledImageArrayNonUniformIndexing = true,
        timelineSemaphore = true,
        bufferDeviceAddress = true,
    }
    next = &vk.PhysicalDeviceVulkan13Features {
        sType = .PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        pNext = next,
        dynamicRendering = true,
        synchronization2 = true,
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
    next = &vk.PhysicalDeviceAccelerationStructureFeaturesKHR {
        sType = .PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        pNext = next,
        accelerationStructure = true,
    }
    next = &vk.PhysicalDeviceRayTracingPipelineFeaturesKHR {
        sType = .PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        pNext = next,
        rayTracingPipeline = true,
    }
    next = &vk.PhysicalDeviceFeatures2 {
        sType = .PHYSICAL_DEVICE_FEATURES_2,
        pNext = next,
        features = {
            geometryShader = true,  // For the tri_idx gbuffer.
        }
    }
    next = &vk.PhysicalDeviceRayTracingPositionFetchFeaturesKHR {
        sType = .PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
        pNext = next,
        rayTracingPositionFetch = true,
    }

    device_ci := vk.DeviceCreateInfo {
        sType = .DEVICE_CREATE_INFO,
        pNext = next,
        queueCreateInfoCount = u32(len(queue_create_infos)),
        pQueueCreateInfos = raw_data(queue_create_infos),
        enabledExtensionCount = u32(len(device_extensions)),
        ppEnabledExtensionNames = raw_data(device_extensions),
    }
    vk_check(vk.CreateDevice(chosen_phys_device, &device_ci, nil, &res.device))

    vk.load_proc_addresses_device(res.device)
    if vk.BeginCommandBuffer == nil do fatal_error("Failed to load Vulkan device API")

    vk.GetDeviceQueue(res.device, queue_family_idx, 0, &res.queue)
    vk.GetDeviceQueue(res.device, queue_family_idx, 1, &res.lm_queue)
    return res
}
