
package main

import log "core:log"
import "core:fmt"

import "../../gpu"

import sdl "vendor:sdl3"

Window_Size_X :: 1000
Window_Size_Y :: 1000
Frames_In_Flight :: 3
Example_Name :: "Textures"

main :: proc()
{
    fmt.println("Work in Progress!")

    ok_i := sdl.Init({ .VIDEO })
    assert(ok_i)

    console_logger := log.create_console_logger()
    defer log.destroy_console_logger(console_logger)
    context.logger = console_logger

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

    texture_desc_heap := gpu.mem_alloc_typed(gpu.Texture_Descriptor, 65536)
    defer gpu.mem_free(raw_data(texture_desc_heap))

    Vertex :: struct { pos: [4]f32 }

    arena := gpu.arena_init(1024 * 1024)
    defer gpu.arena_destroy(&arena)

    verts := gpu.arena_alloc_array(&arena, Vertex, 4)
    verts.cpu[0].pos = { -0.5,  0.5, 0.0, 0.0 }
    verts.cpu[1].pos = {  0.5, -0.5, 0.0, 0.0 }
    verts.cpu[2].pos = {  0.5,  0.5, 0.0, 0.0 }
    verts.cpu[3].pos = { -0.5, -0.5, 0.0, 0.0 }

    indices := gpu.arena_alloc_array(&arena, u32, 6)
    indices.cpu[0] = 0
    indices.cpu[1] = 1
    indices.cpu[2] = 2
    indices.cpu[3] = 0
    indices.cpu[4] = 1
    indices.cpu[5] = 3

    verts_local := gpu.mem_alloc_typed_gpu(Vertex, 4)
    indices_local := gpu.mem_alloc_typed_gpu(u32, 6)
    defer {
        gpu.mem_free(verts_local)
        gpu.mem_free(indices_local)
    }

    tex_desc := gpu.Texture_Desc {
        dimensions = { 100, 100, 1 },
        mip_count = 1,
        layer_count = 1,
        sample_count = 1,
        format = .RGBA8_Unorm,
        usage = { .Sampled },
    }
    tex_size, tex_align := gpu.texture_size_and_align(tex_desc)
    tex_ptr := gpu.mem_alloc(tex_size, tex_align, .GPU)
    defer gpu.mem_free(tex_ptr)
    texture := gpu.texture_create(tex_desc, tex_ptr)
    defer gpu.texture_destroy(&texture)

    queue := gpu.get_queue()

    upload_cmd_buf := gpu.commands_begin(queue)
    gpu.cmd_mem_copy(upload_cmd_buf, verts.gpu, verts_local, u64(len(verts.cpu)) * size_of(verts.cpu[0]))
    gpu.cmd_mem_copy(upload_cmd_buf, indices.gpu, indices_local, u64(len(indices.cpu)) * size_of(indices.cpu[0]))
    gpu.cmd_barrier(upload_cmd_buf, .Transfer, .All, {})
    gpu.queue_submit(queue, { upload_cmd_buf })

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

        frame_arena := &frame_arenas[next_frame % Frames_In_Flight]

        swapchain := gpu.swapchain_acquire_next()  // Blocks CPU until at least one frame is available.

        cmd_buf := gpu.commands_begin(queue)
        gpu.cmd_begin_render_pass(cmd_buf, {
            color_attachments = {
                { view = swapchain, clear_color = { 0.7, 0.7, 0.7, 1.0 } }
            }
        })
        gpu.cmd_set_shaders(cmd_buf, vert_shader, frag_shader)
        Vert_Data :: struct {
            verts: rawptr,
        }
        verts_data := gpu.arena_alloc(frame_arena, Vert_Data)
        verts_data.cpu.verts = verts_local

        gpu.cmd_draw_indexed_instanced(cmd_buf, verts_data.gpu, nil, indices_local, u32(len(indices.cpu)), 1)
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
