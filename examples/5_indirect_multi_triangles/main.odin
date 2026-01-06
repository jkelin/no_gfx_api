
package main

import log "core:log"
import "core:math"
import "core:math/linalg"

import "../../gpu"

import sdl "vendor:sdl3"

Window_Size_X :: 1000
Window_Size_Y :: 1000
Frames_In_Flight :: 3
Example_Name :: "Indirect Multi Triangles"
Num_Triangles :: 32

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

    vert_shader := gpu.shader_create(#load("shaders/test.vert.spv", []u32), .Vertex)
    frag_shader := gpu.shader_create(#load("shaders/test.frag.spv", []u32), .Fragment)
    defer {
        gpu.shader_destroy(&vert_shader)
        gpu.shader_destroy(&frag_shader)
    }

    Vertex :: struct { pos: [4]f32 }

    arena := gpu.arena_init(1024 * 1024)
    defer gpu.arena_destroy(&arena)

    verts := gpu.arena_alloc_array(&arena, Vertex, 3)
    verts.cpu[0].pos = { -0.5,  0.5, 0.0, 0.0 }
    verts.cpu[1].pos = {  0.0, -0.5, 0.0, 0.0 }
    verts.cpu[2].pos = {  0.5,  0.5, 0.0, 0.0 }

    indices := gpu.arena_alloc_array(&arena, u32, 3)
    indices.cpu[0] = 0
    indices.cpu[1] = 2
    indices.cpu[2] = 1

    verts_local := gpu.mem_alloc_typed_gpu(Vertex, 3)
    indices_local := gpu.mem_alloc_typed_gpu(u32, 3)
    
    indirect_command_cpu_mem := gpu.mem_alloc_typed(gpu.Draw_Indexed_Indirect_Command, Num_Triangles)
    defer gpu.mem_free_typed(indirect_command_cpu_mem)

    count := gpu.arena_alloc_array(&arena, u32, 1)
    count.cpu[0] = Num_Triangles

    count_local := gpu.mem_alloc_typed_gpu(u32, 1)

    indirect_command := gpu.host_to_device_ptr(raw_data(indirect_command_cpu_mem))

    Indirect_VertData :: struct {
        color: [4]f32,  // 16 bytes, offset 0
        pos: [4]f32,     // 16 bytes, offset 16
        size: f32,       // 4 bytes, offset 32
        _padding: [3]f32, // 12 bytes padding to match std140 alignment (struct size = 48 bytes)
    }

    indirect_vert_data := gpu.arena_alloc_array(&arena, Indirect_VertData, Num_Triangles)
    
    // Arrange triangles in a circle
    circle_radius: f32 = 0.6
    for i in 0..<Num_Triangles {
        angle := f32(i) / f32(Num_Triangles) * math.PI * 2.0
        
        // Position on circle
        x := math.cos(angle) * circle_radius
        y := math.sin(angle) * circle_radius
        
        // HSL color: hue varies around the circle (0-360 degrees), saturation and lightness fixed
        hue := angle / (math.PI * 2.0)  // 0.0 to 1.0
        saturation: f32 = 1.0
        lightness: f32 = 0.5
        
        // Convert HSL to RGB
        rgb := hsl_to_rgb(hue, saturation, lightness)
        
        indirect_vert_data.cpu[i].color = { rgb.x, rgb.y, rgb.z, 1.0 }
        indirect_vert_data.cpu[i].pos = { x, y, 0.0, 0.0 }
        indirect_vert_data.cpu[i].size = 0.1

        indirect_command_cpu_mem[i] = gpu.Draw_Indexed_Indirect_Command {
            index_count = 3,
            instance_count = 1,
            first_index = 0,
            vertex_offset = 0,
            first_instance = 0,
        }
    }

    indirect_vert_data_local := gpu.mem_alloc_typed_gpu(Indirect_VertData, Num_Triangles)

    defer {
        gpu.mem_free(verts_local)
        gpu.mem_free(indices_local)
        gpu.mem_free(count_local)
        gpu.mem_free(indirect_vert_data_local)
    }

    queue := gpu.get_queue()

    upload_cmd_buf := gpu.commands_begin(queue)
    gpu.cmd_mem_copy(upload_cmd_buf, verts.gpu, verts_local, 3 * size_of(Vertex))
    gpu.cmd_mem_copy(upload_cmd_buf, indices.gpu, indices_local, 3 * size_of(u32))
    gpu.cmd_mem_copy(upload_cmd_buf, count.gpu, count_local, 1 * size_of(u32))
    gpu.cmd_mem_copy(upload_cmd_buf, indirect_vert_data.gpu, indirect_vert_data_local, Num_Triangles * size_of(Indirect_VertData))
    gpu.cmd_barrier(upload_cmd_buf, .Transfer, .All, {})
    gpu.queue_submit(queue, { upload_cmd_buf })

    now_ts := sdl.GetPerformanceCounter()

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

        frame_arena := &frame_arenas[next_frame % Frames_In_Flight]

        swapchain := gpu.swapchain_acquire_next()  // Blocks CPU until at least one frame is available.

        cmd_buf := gpu.commands_begin(queue)
        gpu.cmd_begin_render_pass(cmd_buf, {
            color_attachments = {
                { view = swapchain, clear_color = changing_color(delta_time) }
            }
        })
        gpu.cmd_set_shaders(cmd_buf, vert_shader, frag_shader)
        Vert_Data :: struct {
            verts: rawptr,
        }
        shared_vert_data := gpu.arena_alloc(frame_arena, Vert_Data)
        shared_vert_data.cpu.verts = verts_local

        // Draw multiple indexed triangles using indirect rendering
        // Arguments:
        //   cmd_buf: Command buffer to record the draw command
        //   indirect_vert_data_local: GPU pointer to array of Indirect_VertData (per-draw data: color, pos, size)
        //   nil: GPU pointer to per-draw fragment shader data (not used in this example)
        //   indices_local: GPU pointer to index buffer (u32 array)
        //   indirect_command: GPU pointer to array of VkDrawIndexedIndirectCommand (draw parameters)
        //   count_local: GPU pointer to u32 containing the number of draws to execute
        //   shared_vert_data.gpu: GPU pointer to shared vertex data (used by all draws - the triangle vertices)
        //   nil: GPU pointer to shared fragment shader data (not used in this example)
        gpu.cmd_draw_indexed_instanced_indirect_multi_data(cmd_buf, indirect_vert_data_local, nil, indices_local, indirect_command, count_local, shared_vert_data.gpu, nil)
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

changing_color :: proc(delta_time: f32) -> [4]f32
{
    @(static) t: f32
    t = math.mod(t + delta_time * 1.7, math.PI * 2)

    color_a := [4]f32 { 0.2, 0.2, 0.2, 1.0 }
    color_b := [4]f32 { 0.4, 0.4, 0.4, 1.0 }
    return linalg.lerp(color_a, color_b, math.sin(t) * 0.5 + 0.5)
}

// Convert HSL to RGB (hue: 0-1, saturation: 0-1, lightness: 0-1)
hsl_to_rgb :: proc(h: f32, s: f32, l: f32) -> linalg.Vector3f32
{
    c := (1.0 - abs(2.0 * l - 1.0)) * s
    x := c * (1.0 - abs(math.mod(h * 6.0, 2.0) - 1.0))
    m := l - c / 2.0
    
    r, g, b: f32
    
    if h < 1.0/6.0 {
        r, g, b = c, x, 0.0
    } else if h < 2.0/6.0 {
        r, g, b = x, c, 0.0
    } else if h < 3.0/6.0 {
        r, g, b = 0.0, c, x
    } else if h < 4.0/6.0 {
        r, g, b = 0.0, x, c
    } else if h < 5.0/6.0 {
        r, g, b = x, 0.0, c
    } else {
        r, g, b = c, 0.0, x
    }
    
    return { r + m, g + m, b + m }
}
