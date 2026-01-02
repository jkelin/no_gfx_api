
package main

import "core:fmt"
import vmem "core:mem/virtual"
import "core:strings"
import "base:runtime"
import "core:os/os2"

Shader_Type :: enum
{
    Vertex,
    Fragment
}

codegen :: proc(ast: Ast, shader_type: Shader_Type, input_path: string, output_path: string)
{
    write_preamble()

    arena_backing: vmem.Arena
    ok_a := vmem.arena_init_growing(&arena_backing)
    assert(ok_a == nil)
    codegen_arena := vmem.arena_allocator(&arena_backing)
    defer free_all(codegen_arena)

    context.allocator = codegen_arena

    for loc, &type in ast.used_out_locations {
        writefln("layout(location = %v) out %v _res_out_loc%v_;", loc, type_to_glsl(&type), loc)
    }
    for loc, &type in ast.used_in_locations {
        writefln("layout(location = %v) in %v _res_in_loc%v_;", loc, type_to_glsl(&type), loc)
    }

    writeln("")

    writefln("layout(buffer_reference) readonly buffer _res_ptr_void;")
    for &type in ast.used_types
    {
        if type.kind == .Pointer || type.kind == .Slice {
            writefln("layout(buffer_reference) readonly buffer %v;", type_to_glsl(&type))
        }
    }

    writeln("")

    // Generate all decls (structs are already defined here because you can't forward-declare structs in GLSL)
    for decl in ast.scope.decls
    {
        switch decl.type.kind
        {
            case .Poison: {}
            case .Label: {}
            case .Pointer: {}
            case .Slice: {}
            case .Primitive: {}
            case .Struct:
            {
                writefln("struct %v", decl.name)
                writeln("{")
                if writer_scope()
                {
                    for field in decl.type.members
                    {
                        writefln("%v %v;", type_to_glsl(field.type), field.name)
                    }
                }
                writeln("};")
                writeln("")
            }
            case .Proc:
            {
                is_main := decl.name == "main"

                write_begin("")
                ret_type_glsl := "void" if is_main else type_to_glsl(decl.type.ret)
                writef("%v %v(", ret_type_glsl, decl.name)
                for arg, i in decl.type.args
                {
                    if arg.attr != nil do continue

                    writef("%v %v", type_to_glsl(arg.type), arg.name)
                    if i < len(decl.type.args) - 1 {
                        write(", ")
                    }
                }
                writeln(");")
            }
        }
    }

    writefln("layout(buffer_reference, std140) readonly buffer _res_ptr_void {{ uint _res_void_; }};")
    for &type in ast.used_types
    {
        if type.kind == .Pointer || type.kind == .Slice {
            writefln("layout(buffer_reference, std140) readonly buffer %v {{ %v _res_; }};", type_to_glsl(&type), type_to_glsl(type.base))
        }
    }

    writeln("")

    // Generate bindings
    writeln("layout(set = 0, binding = 0) uniform texture2D _res_textures_[];")
    writeln("layout(set = 1, binding = 0) uniform writeonly image2D _res_textures_rw_[];")
    writeln("layout(set = 2, binding = 0) uniform sampler _res_samplers_[];")
    writeln("")

    writeln("layout(push_constant, std140) uniform Push")
    writeln("{")
    data_type_str := type_to_glsl(ast.used_data_type) if ast.used_data_type != nil else "_res_ptr_void"
    if writer_scope()
    {
        if shader_type == .Vertex {
            writefln("%v _res_data_;", data_type_str)
        } else {
            writefln("%v _res_vert_data_;", data_type_str)
        }

        if shader_type == .Fragment {
            writefln("%v _res_data_;", data_type_str)
        } else {
            writefln("%v _res_frag_data_;", data_type_str)
        }
    }
    writeln("};")
    writeln("")

    for proc_def in ast.procs
    {
        decl := proc_def.decl
        is_main := decl.name == "main"

        write_begin("")
        ret_type_glsl := "void" if is_main else type_to_glsl(decl.type.ret)
        writef("%v %v(", ret_type_glsl, decl.name)
        for arg, i in decl.type.args
        {
            if arg.attr != nil do continue

            writef("%v %v", type_to_glsl(arg.type), arg.name)
            if i < len(decl.type.args) - 1 {
                write(", ")
            }
        }
        writeln(")")
        writeln("{")
        if writer_scope()
        {
            // Declare all variables
            for var_decl in proc_def.scope.decls
            {
                if var_decl.attr == nil {
                    writefln("%v %v;", type_to_glsl(var_decl.type), var_decl.name)
                } else {
                    writefln("%v %v = %v;", type_to_glsl(var_decl.type), var_decl.name, attribute_to_glsl(var_decl.attr.?))
                }
            }

            for statement in proc_def.statements
            {
                codegen_statement(statement, ast, proc_def)
            }
        }
        writeln("}")
        writeln("")
    }

    writer_output_to_file(output_path)
}

codegen_statement :: proc(statement: ^Ast_Statement, ast: Ast, proc_def: ^Ast_Proc_Def)
{
    write_begin("")

    decl := proc_def.decl
    is_main := decl.name == "main"
    ret_attr := decl.type.ret_attr

    switch stmt in statement.derived_statement
    {
        case ^Ast_Stmt_Expr:
        {
            codegen_expr(stmt.expr)
        }
        case ^Ast_Assign:
        {
            codegen_expr(stmt.lhs)
            write(" = ")
            codegen_expr(stmt.rhs)
        }
        case ^Ast_Return:
        {
            if is_main
            {
                type := stmt.expr.type
                if type.kind == .Label do type = type_get_base(type)

                if type.kind == .Struct
                {
                    for member in type.members
                    {
                        if member.attr == nil do continue
                        writef("%v = ", attribute_to_glsl(member.attr.?))
                        codegen_expr(stmt.expr)
                        writef(".%v; ", member.name)
                    }
                }
                else
                {
                    if ret_attr != nil && ret_attr.?.type == .Out_Loc
                    {
                        writef("%v = ", attribute_to_glsl(ret_attr.?))
                        codegen_expr(stmt.expr)
                    }
                    else
                    {
                        panic("Not implemented!")
                    }
                }
            }
            else
            {
                write("return ")
                codegen_expr(stmt.expr)
            }
        }
    }

    write(";\n")
}

codegen_expr :: proc(expression: ^Ast_Expr)
{
    switch expr in expression.derived_expr
    {
        case ^Ast_Binary_Expr:
        {
            codegen_expr(expr.lhs)
            write(expr.token.text)
            codegen_expr(expr.rhs)
        }
        case ^Ast_Ident_Expr:
        {
            write(expr.token.text)
        }
        case ^Ast_Lit_Expr:
        {
            write(expr.token.text)
        }
        case ^Ast_Member_Access:
        {
            codegen_expr(expr.target)
            if expr.target.type.kind == .Pointer || expr.target.type.kind == .Slice {
                writef("._res_.%v", expr.member_name)
            } else {
                writef(".%v", expr.member_name)
            }
        }
        case ^Ast_Array_Access:
        {
            codegen_expr(expr.target)
            write("[")
            codegen_expr(expr.idx_expr)
            write("]._res_")
        }
        case ^Ast_Call:
        {
            // Check for intrinsics
            is_intrinsic := false
            call_ident, is_ident := expr.target.derived_expr.(^Ast_Ident_Expr)
            if is_ident
            {
                text := call_ident.token.text
                if text == "sample"
                {
                    assert(len(expr.args) == 3)

                    write("texture(sampler2D(_res_textures_[nonuniformEXT(")
                    codegen_expr(expr.args[0])
                    write(")], _res_samplers_[nonuniformEXT(")
                    codegen_expr(expr.args[1])
                    write(")]), ")
                    codegen_expr(expr.args[2])
                    write(")")

                    is_intrinsic = true
                }
            }

            if is_intrinsic do break

            codegen_expr(expr.target)
            write("(")
            for arg, i in expr.args
            {
                codegen_expr(arg)
                if i < len(expr.args) - 1 {
                    write(", ")
                }
            }
            write(")")
        }
    }
}

type_to_glsl :: proc(type: ^Ast_Type) -> string
{
    if type == nil do return "void"

    switch type.kind
    {
        case .Poison: return "<POISON>"
        case .Label: return type.name.text
        case .Pointer: return strings.concatenate({ "_res_ptr_", type_to_glsl(type.base) })
        case .Slice: return strings.concatenate({ "_res_slice_", type_to_glsl(type.base) })
        case .Proc: assert(false, "Not implemented.")
        case .Struct: assert(false, "Not implemented.")
        case .Primitive:
        {
            switch type.primitive_kind
            {
                case .None: return "NONE"
                case .Float: return "float"
                case .Uint: return "uint"
                case .Int: return "int"
                case .Vec2: return "vec2"
                case .Vec3: return "vec3"
                case .Vec4: return "vec4"
                case .Texture_ID: return "uint"
                case .Sampler_ID: return "uint"
            }
        }
    }
    return ""
}

attribute_to_glsl :: proc(attribute: Ast_Attribute) -> string
{
    val_str := runtime.cstring_to_string(fmt.caprint(attribute.arg, allocator = context.allocator))

    switch attribute.type
    {
        case .Vert_ID:  return "gl_VertexIndex"
        case .Position: return "gl_Position"
        case .Data:     return "_res_data_"
        case .Out_Loc:  return strings.concatenate({"_res_out_loc", val_str, "_"})
        case .In_Loc:   return strings.concatenate({"_res_in_loc", val_str, "_"})
    }

    return {}
}

Writer :: struct
{
    indentation: u32,
    builder: strings.Builder,
}

@(private="file")
writer: Writer

@(deferred_in = writer_scope_end)
writer_scope :: proc() -> bool
{
    writer_scope_begin()
    return true
}

@(private="file")
writer_scope_begin :: proc()
{
    writer.indentation += 1
}

@(private="file")
writer_scope_end :: proc()
{
    writer.indentation -= 1
}

@(private="file")
write_preamble :: proc()
{
    writeln("#version 460")
    writeln("#extension GL_EXT_buffer_reference : require")
    writeln("#extension GL_EXT_buffer_reference2 : require")
    writeln("#extension GL_EXT_nonuniform_qualifier : require")
    writeln("")
}

@(private="file")
writefln :: proc(fmt_str: string, args: ..any)
{
    write_indentation()
    fmt.sbprintfln(&writer.builder, fmt_str, ..args)
}

@(private="file")
writef :: proc(fmt_str: string, args: ..any)
{
    fmt.sbprintf(&writer.builder, fmt_str, ..args)
}

@(private="file")
writeln :: proc(strings: ..any)
{
    write_indentation()
    fmt.sbprintln(&writer.builder, ..strings)
}

@(private="file")
write_begin :: proc(strings: ..any)
{
    write_indentation()
    fmt.sbprint(&writer.builder, ..strings)
}

@(private="file")
write :: proc(strings: ..any)
{
    fmt.sbprint(&writer.builder, ..strings)
}

@(private="file")
write_indentation :: proc()
{
    for _ in 0..<4*writer.indentation {
        fmt.sbprint(&writer.builder, " ")
    }
}

@(private="file")
writer_output_to_file :: proc(path: string)
{
    err := os2.write_entire_file_from_string(path, strings.to_string(writer.builder))
    ensure(err == nil)
}
