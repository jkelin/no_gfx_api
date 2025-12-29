
package main

import "core:fmt"
import "core:os"
import "core:mem"
import vmem "core:mem/virtual"
import str "core:strings"
import "base:runtime"
import fp "core:path/filepath"

main :: proc()
{
    if len(os.args) != 2
    {
        fmt.println("Incorrect Usage. Try: gpu_compiler *.musl")
        return
    }

    init_scratch_arenas()

    perm_arena_backing: vmem.Arena
    ok_a := vmem.arena_init_growing(&perm_arena_backing)
    assert(ok_a == nil)
    perm_arena := vmem.arena_allocator(&perm_arena_backing)
    defer free_all(perm_arena)

    path := os.args[1]
    shader_type_str := fp.ext(fp.stem(path))
    shader_type: Shader_Type
    if shader_type_str == ".vert" {
        shader_type = .Vertex
    } else if shader_type_str == ".frag" {
        shader_type = .Fragment
    } else {
        fmt.println("Could not infer shader type. Try '*.vert.musl' or '*.frag.musl'.")
        return
    }

    output_path := str.concatenate({ fp.stem(path), ".glsl" }, allocator = perm_arena)

    file_content, ok := load_file_and_null_terminate(path, allocator = perm_arena)
    if !ok
    {
        fmt.println("Error: Failed to read file.")
        return
    }

    tokens := lex_file(file_content, allocator = perm_arena)
    ast, ok_p := parse_file(path, tokens, allocator = perm_arena)
    if !ok_p do return
    ok_t := typecheck_ast(ast, allocator = perm_arena)
    if !ok_t do return
    codegen(ast, shader_type, path, output_path)

    fmt.println(path)
}

load_file_and_null_terminate :: proc(path: string, allocator: runtime.Allocator) -> ([]u8, bool)
{
    file_content, ok := os.read_entire_file_from_filename(path)
    if !ok do return {}, false
    defer delete(file_content)

    file_content_null_term := make([]u8, len(file_content) + 1, allocator = allocator)
    copy(file_content_null_term[:], file_content[:])
    file_content_null_term[len(file_content)] = 0
    return file_content_null_term, true
}

// Scratch arenas

scratch_arenas: [4]vmem.Arena

init_scratch_arenas :: proc()
{
    for &scratch in scratch_arenas
    {
        error := vmem.arena_init_growing(&scratch)
        assert(error == nil)
    }
}

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

release_scratch :: #force_inline proc(allocator: mem.Allocator, temp: vmem.Arena_Temp)
{
    vmem.arena_temp_end(temp)
}
