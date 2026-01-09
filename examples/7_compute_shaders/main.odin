
package main

import log "core:log"
import "core:math"
import "core:math/linalg"

import "../../gpu"

import sdl "vendor:sdl3"

Window_Size_X :: 1000
Window_Size_Y :: 1000
Frames_In_Flight :: 3
Example_Name :: "Compute Shader"

// Whether to use indirect dispatch (group counts) or direct dispatch (thread counts)
Use_Indirect :: true

main :: proc()
{
    ok_i := sdl.Init({ .VIDEO })
    assert(ok_i)

    console_logger := log.create_console_logger()
    defer log.destroy_console_logger(console_logger)
    context.logger = console_logger

    ts_freq := sdl.GetPerformanceFrequency()
    max_delta_time: f32 = 1.0 / 10.0  // 10fps

    window_flags :: sdl.WindowFlags {
        .HIGH_PIXEL_DENSITY,
        .VULKAN,
    }
    window := sdl.CreateWindow(Example_Name, Window_Size_X, Window_Size_Y, window_flags)
    ensure(window != nil)

    gpu.init(window, Frames_In_Flight)
    defer gpu.cleanup()

    group_size_x := u32(8)
    group_size_y := u32(8)
    compute_shader := gpu.shader_create_compute(#load("shaders/test.comp.spv", []u32), group_size_x, group_size_y, 1)
    vert_shader := gpu.shader_create(#load("shaders/test.vert.spv", []u32), .Vertex)
    frag_shader := gpu.shader_create(#load("shaders/test.frag.spv", []u32), .Fragment)
    defer {
        gpu.shader_destroy(&compute_shader)
        gpu.shader_destroy(&vert_shader)
        gpu.shader_destroy(&frag_shader)
    }

    // Create a texture for the compute shader to write to
    output_texture := gpu.alloc_and_create_texture({
        type = .D2,
        dimensions = { Window_Size_X, Window_Size_Y, 1 },
        mip_count = 1,
        layer_count = 1,
        sample_count = 1,
        format = .RGBA8_Unorm,
        usage = { .Storage, .Sampled },
    })
    defer gpu.free_and_destroy_texture(&output_texture)

    // Create texture descriptor for RW access (compute shader)
    texture_rw_desc := gpu.texture_rw_view_descriptor(output_texture, {})

    texture_id := u32(0)
    sampler_id := u32(0)
    
    // Allocate texture heap for compute shader
    texture_rw_heap_size := gpu.get_texture_rw_view_descriptor_size()
    texture_rw_heap := gpu.mem_alloc(u64(texture_rw_heap_size))
    defer gpu.mem_free(texture_rw_heap)
    gpu.set_texture_rw_desc(texture_rw_heap, texture_id, texture_rw_desc)
    texture_rw_heap_gpu := gpu.host_to_device_ptr(texture_rw_heap)

    // Create texture descriptor for sampled access (fragment shader)
    texture_desc := gpu.texture_view_descriptor(output_texture, { format = .RGBA8_Unorm })
    
    // Allocate texture heap for fragment shader
    texture_heap := gpu.mem_alloc(size_of(gpu.Texture_Descriptor) * 65536)
    defer gpu.mem_free(texture_heap)
    gpu.set_texture_desc(texture_heap, texture_id, texture_desc)

    // Create sampler
    sampler_heap := gpu.mem_alloc(size_of(gpu.Sampler_Descriptor) * 10)
    defer gpu.mem_free(sampler_heap)
    gpu.set_sampler_desc(sampler_heap, sampler_id, gpu.sampler_descriptor({}))

    // Indirect dispatch command (group counts)
    indirect_dispatch_command_ptr: rawptr
    indirect_dispatch_command_cpu_mem: []gpu.Dispatch_Indirect_Command
    indirect_dispatch_command_cpu_mem = gpu.mem_alloc_typed(gpu.Dispatch_Indirect_Command, 1)
    indirect_dispatch_command_ptr = gpu.host_to_device_ptr(raw_data(indirect_dispatch_command_cpu_mem))
    defer gpu.mem_free_typed(indirect_dispatch_command_cpu_mem)

    Compute_Data :: struct {
        output_texture_id: u32,
        resolution: [2]f32,
        time: f32,
    }

    Vertex :: struct { pos: [3]f32, uv: [2]f32 }

    arena := gpu.arena_init(1024 * 1024)
    defer gpu.arena_destroy(&arena)

    // Create fullscreen quad
    verts := gpu.arena_alloc_array(&arena, Vertex, 4)
    verts.cpu[0].pos = { -1.0,  1.0, 0.0 }  // Top-left
    verts.cpu[1].pos = {  1.0, -1.0, 0.0 }  // Bottom-right
    verts.cpu[2].pos = {  1.0,  1.0, 0.0 }  // Top-right
    verts.cpu[3].pos = { -1.0, -1.0, 0.0 }  // Bottom-left
    verts.cpu[0].uv  = {  0.0,  0.0 }
    verts.cpu[1].uv  = {  1.0,  1.0 }
    verts.cpu[2].uv  = {  1.0,  0.0 }
    verts.cpu[3].uv  = {  0.0,  1.0 }

    indices := gpu.arena_alloc_array(&arena, u32, 6)
    indices.cpu[0] = 0
    indices.cpu[1] = 2
    indices.cpu[2] = 1
    indices.cpu[3] = 0
    indices.cpu[4] = 1
    indices.cpu[5] = 3

    verts_local := gpu.mem_alloc_typed_gpu(Vertex, 4)
    indices_local := gpu.mem_alloc_typed_gpu(u32, 6)
    defer {
        gpu.mem_free(verts_local)
        gpu.mem_free(indices_local)
    }

    queue := gpu.get_queue()

    upload_cmd_buf := gpu.commands_begin(queue)
    gpu.cmd_mem_copy(upload_cmd_buf, verts.gpu, verts_local, u64(len(verts.cpu)) * size_of(verts.cpu[0]))
    gpu.cmd_mem_copy(upload_cmd_buf, indices.gpu, indices_local, u64(len(indices.cpu)) * size_of(indices.cpu[0]))
    gpu.cmd_barrier(upload_cmd_buf, .Transfer, .All, {})
    gpu.queue_submit(queue, { upload_cmd_buf })

    now_ts := sdl.GetPerformanceCounter()
    total_time: f32 = 0.0

    frame_arenas: [Frames_In_Flight]gpu.Arena
    for &frame_arena in frame_arenas do frame_arena = gpu.arena_init(1024 * 1024)
    defer for &frame_arena in frame_arenas do gpu.arena_destroy(&frame_arena)
    next_frame := u64(1)
    frame_sem := gpu.semaphore_create(0)
    defer gpu.semaphore_destroy(&frame_sem)
    for true
    {
        proceed := handle_window_events(window)
        if !proceed do break
        if .MINIMIZED in sdl.GetWindowFlags(window)
        {
            sdl.Delay(16)
            continue
        }

        if next_frame > Frames_In_Flight {
            gpu.semaphore_wait(frame_sem, next_frame - Frames_In_Flight)
        }

        last_ts := now_ts
        now_ts = sdl.GetPerformanceCounter()
        delta_time := min(max_delta_time, f32(f64((now_ts - last_ts)*1000) / f64(ts_freq)) / 1000.0)
        total_time += delta_time

        frame_arena := &frame_arenas[next_frame % Frames_In_Flight]

        swapchain := gpu.swapchain_acquire_next()  // Blocks CPU until at least one frame is available.

        // Allocate compute data for this frame with current time and resolution
        compute_data := gpu.arena_alloc(frame_arena, Compute_Data)
        compute_data.cpu.output_texture_id = texture_id
        compute_data.cpu.resolution = { f32(Window_Size_X), f32(Window_Size_Y) }
        compute_data.cpu.time = total_time

        cmd_buf := gpu.commands_begin(queue)
        
        // Dispatch compute shader to write to texture
        gpu.cmd_set_texture_heap(cmd_buf, nil, texture_rw_heap_gpu, nil)
        gpu.cmd_set_compute_shader(cmd_buf, compute_shader)
        
        if Use_Indirect {
            // Calculate group counts from thread counts
            // Indirect compute shaders do not support dispatching by number of threads
            indirect_dispatch_command_cpu_mem[0] = gpu.Dispatch_Indirect_Command {
                num_groups_x = (Window_Size_X + group_size_x - 1) / group_size_x,
                num_groups_y = (Window_Size_Y + group_size_y - 1) / group_size_y,
                num_groups_z = u32(1),
            }

            gpu.cmd_dispatch_indirect(cmd_buf, compute_data.gpu, indirect_dispatch_command_ptr)
        } else {
            // Use cmd_dispatch_threads to specify thread counts (converts to groups automatically)
            gpu.cmd_dispatch_threads(cmd_buf, compute_data.gpu, Window_Size_X, Window_Size_Y, 1)
        }
        
        // Barrier to ensure compute shader finishes before rendering
        gpu.cmd_barrier(cmd_buf, .Compute, .Fragment_Shader, {})
        
        // Render the texture to the swapchain using a fullscreen quad
        gpu.cmd_begin_render_pass(cmd_buf, {
            color_attachments = {
                { texture = swapchain, clear_color = { 0.0, 0.0, 0.0, 1.0 } }
            }
        })
        gpu.cmd_set_shaders(cmd_buf, vert_shader, frag_shader)
        textures := gpu.host_to_device_ptr(texture_heap)
        samplers := gpu.host_to_device_ptr(sampler_heap)
        gpu.cmd_set_texture_heap(cmd_buf, textures, nil, samplers)
        
        Vert_Data :: struct {
            verts: rawptr,
        }
        verts_data := gpu.arena_alloc(frame_arena, Vert_Data)
        verts_data.cpu.verts = verts_local
        
        Frag_Data :: struct {
            texture_id: u32,
            sampler_id: u32,
        }
        frag_data := gpu.arena_alloc(frame_arena, Frag_Data)
        frag_data.cpu.texture_id = texture_id
        frag_data.cpu.sampler_id = sampler_id
        
        gpu.cmd_draw_indexed_instanced(cmd_buf, verts_data.gpu, frag_data.gpu, indices_local, u32(len(indices.cpu)), 1)
        gpu.cmd_end_render_pass(cmd_buf)
        gpu.queue_submit(queue, { cmd_buf }, frame_sem, next_frame)

        gpu.swapchain_present(queue, frame_sem, next_frame)
        next_frame += 1

        gpu.arena_free_all(frame_arena)
    }

    gpu.wait_idle()
}

handle_window_events :: proc(window: ^sdl.Window) -> (proceed: bool)
{
    event: sdl.Event
    proceed = true
    for sdl.PollEvent(&event)
    {
        #partial switch event.type
        {
            case .QUIT:
                proceed = false
            case .WINDOW_CLOSE_REQUESTED:
            {
                if event.window.windowID == sdl.GetWindowID(window) {
                    proceed = false
                }
            }
        }
    }

    return
}