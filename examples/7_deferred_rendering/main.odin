
package main

import log "core:log"
import "../../gpu"
import "core:math"
import "core:math/linalg"
import "base:runtime"
import "core:fmt"
import "core:os"
import intr "base:intrinsics"

import sdl "vendor:sdl3"

import shared "../shared"

Window_Size_X :: 1000
Window_Size_Y :: 1000
Frames_In_Flight :: 3
Example_Name :: "Deferred Rendering"

// For example purposes, absolute path is baked into the executable which means you can't use it on another machine
Model_Scene :: #directory + "../sample_assets/Models/Sponza/glTF/Sponza.gltf"

Render_Target :: enum {
    Color,
    Normal,
    Depth,
}

current_render_target := Render_Target.Color

main :: proc()
{
    fmt.println("Right-click + WASD for first-person controls.")
    fmt.println("Left-click to change render target.")
    fmt.println("Current render target:", current_render_target)

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

    dim := [3]u32 {  }
    depth_texture := gpu.alloc_and_create_texture({
        dimensions = { Window_Size_X, Window_Size_Y, 1 },
        format = .D32_Float,
        mip_count = 1,
        layer_count = 1,
        sample_count = 1,
        usage = { .Depth_Stencil_Attachment },
    })
    defer gpu.free_and_destroy_texture(&depth_texture)

    vert_shader := gpu.shader_create(#load("shaders/test.vert.spv", []u32), .Vertex)
    frag_shader := gpu.shader_create(#load("shaders/test.frag.spv", []u32), .Fragment)
    defer {
        gpu.shader_destroy(&vert_shader)
        gpu.shader_destroy(&frag_shader)
    }

    upload_arena := gpu.arena_init(1024 * 1024 * 1024)
    defer gpu.arena_destroy(&upload_arena)

    queue := gpu.get_queue()

    upload_cmd_buf := gpu.commands_begin(queue)
    scene := shared.load_scene_gltf(Model_Scene, &upload_arena, upload_cmd_buf, true)
    defer shared.destroy_scene(&scene)
    gpu.cmd_barrier(upload_cmd_buf, .Transfer, .All, {})
    gpu.queue_submit(queue, { upload_cmd_buf })
    
    // Set up texture and sampler heaps
    texture_heap := gpu.mem_alloc(size_of(gpu.Texture_Descriptor) * 65536)
    defer gpu.mem_free(texture_heap)
    sampler_heap := gpu.mem_alloc(size_of(gpu.Sampler_Descriptor) * 10)
    defer gpu.mem_free(sampler_heap)
    
    // Upload textures to texture heap
    for tex, idx in scene.textures {
        gpu.set_texture_desc(texture_heap, u32(idx), gpu.texture_view_descriptor(tex, { format = .RGBA8_Unorm }))
    }
    
    // Create default sampler
    gpu.set_sampler_desc(sampler_heap, 0, gpu.sampler_descriptor({}))

    now_ts := sdl.GetPerformanceCounter()

    frame_arenas: [Frames_In_Flight]gpu.Arena
    for &frame_arena in frame_arenas do frame_arena = gpu.arena_init(10 * 1024 * 1024)
    defer for &frame_arena in frame_arenas do gpu.arena_destroy(&frame_arena)
    next_frame := u64(1)
    frame_sem := gpu.semaphore_create(0)
    defer gpu.semaphore_destroy(&frame_sem)
    for true
    {
        proceed := shared.handle_window_events(window)
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

        handle_render_target_input()

        world_to_view := shared.first_person_camera_view(delta_time)
        aspect_ratio := f32(Window_Size_X) / f32(Window_Size_Y)
        view_to_proj := linalg.matrix4_perspective_f32(math.RAD_PER_DEG * 59.0, aspect_ratio, 0.1, 1000.0, false)

        frame_arena := &frame_arenas[next_frame % Frames_In_Flight]

        swapchain := gpu.swapchain_acquire_next()  // Blocks CPU until at least one frame is available.

        cmd_buf := gpu.commands_begin(queue)
        gpu.cmd_begin_render_pass(cmd_buf, {
            color_attachments = {
                { texture = swapchain, clear_color = { 0.7, 0.7, 0.7, 1.0 } }
            },
            depth_attachment = gpu.Render_Attachment {
                texture = depth_texture, clear_color = 1.0
            },
        })
        gpu.cmd_set_shaders(cmd_buf, vert_shader, frag_shader)
        
        // Set texture and sampler heaps
        textures := gpu.host_to_device_ptr(texture_heap)
        samplers := gpu.host_to_device_ptr(sampler_heap)
        gpu.cmd_set_texture_heap(cmd_buf, textures, nil, samplers)

        gpu.cmd_set_depth_state(cmd_buf, { mode = { .Read, .Write }, compare = .Less })

        for instance in scene.instances
        {
            mesh := scene.meshes[instance.mesh_idx]

            Vert_Data :: struct #all_or_none {
                positions: rawptr,
                normals: rawptr,
                uvs: rawptr,
                model_to_world: [16]f32,
                model_to_world_normal: [16]f32,
                world_to_view: [16]f32,
                view_to_proj: [16]f32,
            }
            Frag_Data :: struct #all_or_none {
                albedo_texture_id: u32,
            }
            #assert(size_of(Vert_Data) == 8+8+8+64+64+64+64)
            verts_data := gpu.arena_alloc(frame_arena, Vert_Data)
            verts_data.cpu^ = {
                positions = mesh.pos,
                normals = mesh.normals,
                uvs = mesh.uvs,
                model_to_world = intr.matrix_flatten(instance.transform),
                model_to_world_normal = intr.matrix_flatten(linalg.transpose(linalg.inverse(instance.transform))),
                world_to_view = intr.matrix_flatten(world_to_view),
                view_to_proj = intr.matrix_flatten(view_to_proj),
            }
            
            frags_data := gpu.arena_alloc(frame_arena, Frag_Data)
            albedo_tex_id: u32
            if mesh.albedo_texture_id != nil {
                albedo_tex_id = mesh.albedo_texture_id.?
            } else {
                albedo_tex_id = max(u32)  // Sentinel value for "no texture"
            }
            frags_data.cpu^ = {
                albedo_texture_id = albedo_tex_id,
            }

            gpu.cmd_draw_indexed_instanced(cmd_buf, verts_data.gpu, frags_data.gpu, mesh.indices, mesh.idx_count, 1)
        }

        gpu.cmd_end_render_pass(cmd_buf)
        gpu.queue_submit(queue, { cmd_buf }, frame_sem, next_frame)

        gpu.swapchain_present(queue, frame_sem, next_frame)
        next_frame += 1

        gpu.arena_free_all(frame_arena)
    }

    gpu.wait_idle()
}

handle_render_target_input :: proc() {
    if shared.INPUT.pressed_left_click {
        switch current_render_target {
            case .Color:
                current_render_target = .Normal
            case .Normal:
                current_render_target = .Depth
            case .Depth:
                current_render_target = .Color
        }

        fmt.println("Current render target:", current_render_target)
    }
}