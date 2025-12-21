
package main

import "core:fmt"
import vmem "core:mem/virtual"
import "core:mem"
import "core:strings"
import "base:runtime"

Shader_Type :: enum
{
    Vertex,
    Fragment
}

codegen :: proc(ast: Ast, shader_type: Shader_Type)
{
    write_preamble()

    arena_backing: vmem.Arena
    ok_a := vmem.arena_init_growing(&arena_backing)
    assert(ok_a == nil)
    codegen_arena := vmem.arena_allocator(&arena_backing)
    defer free_all(codegen_arena)

    context.allocator = codegen_arena

    for loc in ast.used_out_locations {
        writefln("layout(location = %v) out _res_out_loc%v_;", loc, loc)
    }
    for loc in ast.used_in_locations {
        writefln("layout(location = %v) in _res_in_loc%v_;", loc, loc)
    }

    writeln("")

    writefln("layout(buffer_reference) readonly buffer _res_ptr_void;")
    for type in ast.used_types
    {
        if type.is_ptr {
            writefln("layout(buffer_reference) readonly buffer _res_ptr_%v;", type.name)
        } else if type.is_slice {
            writefln("layout(buffer_reference) readonly buffer _res_slice_%v;", type.name)
        }
    }

    writeln("")

    writefln("layout(buffer_reference) readonly buffer _res_ptr_void {{ u32 _res_void_; }}")
    for type in ast.used_types
    {
        if type.is_ptr {
            writefln("layout(buffer_reference) readonly buffer _res_ptr_%v {{ %v _res_; }};", type.name, type.name)
        } else if type.is_slice {
            writefln("layout(buffer_reference) readonly buffer _res_slice_%v {{ %v _res_; }};", type.name, type.name)
        }
    }

    writeln("")

    writeln("layout(push_constant, std140) uniform Push")
    writeln("{")
    if writer_scope()
    {
        if shader_type == .Vertex {
            writefln("_res_ptr_%v _res_data_;", ast.used_data_type)
        } else {
            writefln("_res_ptr_%v _res_vert_data_;", ast.used_data_type)
        }

        if shader_type == .Fragment {
            writefln("_res_ptr_%v _res_data_;", ast.used_data_type)
        } else {
            writefln("_res_ptr_%v _res_frag_data_;", ast.used_data_type)
        }
    }
    writeln("}")
    writeln("")

    for declaration in ast.scope.decls
    {
        switch decl in declaration.derived_decl
        {
            case ^Ast_Struct_Decl:
            {
                writefln("struct %v", decl.name)
                writeln("{")
                if writer_scope()
                {
                    for field in decl.fields
                    {
                        writefln("%v %v;", type_to_glsl(field.type), field.name)
                    }
                }
                writeln("};")
                writeln("")
            }
            case ^Ast_Proc_Decl:
            {
                is_main := decl.name == "main"

                write_begin("")
                ret_type_glsl := "void" if is_main else type_to_glsl(decl.return_type)
                writef("%v %v(", type_to_glsl(decl.return_type), decl.name)
                for arg, i in decl.args
                {
                    if arg.attr != nil do continue

                    writef("%v %v", type_to_glsl(arg.type), arg.name)
                    if i < len(decl.args) - 1 {
                        write(", ")
                    }
                }
                writeln(")")
                writeln("{")
                if writer_scope()
                {
                    for arg, i in decl.args
                    {
                        if arg.attr == nil do continue
                        writefln("%v %v = %v;", type_to_glsl(arg.type), arg.name, attribute_to_glsl(arg.attr.?))
                    }

                    for statement in decl.statements
                    {
                        codegen_statement(statement, ast, decl)
                    }
                }
                writeln("}")
                writeln("")
            }
            case:
            {
                fmt.println("Error!")
            }
        }
    }
}

codegen_statement :: proc(statement: ^Ast_Statement, ast: Ast, proc_def: ^Ast_Proc_Decl)
{
    write_begin("")

    is_main := proc_def.name == "main"
    ret_attr := proc_def.return_attr

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
        case ^Ast_Var_Decl:
        {
            writef("%v %v", type_to_glsl(stmt.type), stmt.name)
        }
        case ^Ast_Return:
        {
            if is_main
            {
                #partial switch expr in stmt.expr.derived_expr
                {
                    case ^Ast_Ident_Expr:
                    {
                        info, ok := search_name(ast, proc_def, expr.token.text, expr.token)
                        if ok
                        {
                            if info.is_primitive
                            {
                                writef("%v = ", attribute_to_glsl(info.attr.?))
                                codegen_expr(stmt.expr)
                            }
                        }
                        else
                        {
                            panic("Not implemented!")
                        }
                    }
                    case:
                    {
                        ret_attr_unpacked, ok := ret_attr.?
                        if ok && ret_attr_unpacked.type == .Out_Loc
                        {
                            write("_res_out_loc0_ = ")
                            codegen_expr(stmt.expr)
                        }
                        else
                        {
                            panic("Not implemented!")
                        }
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
            writef(".%v", expr.member_name)
        }
        case ^Ast_Array_Access:
        {
            codegen_expr(expr.target)
            write("[")
            codegen_expr(expr.idx_expr)
            write("]")
        }
        case ^Ast_Call:
        {
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

// TODO: Very prototype-y.
search_type_name_of_name :: proc(proc_def: ^Ast_Proc_Decl, name: string, pos: Token) -> (string, Maybe(Ast_Attribute), bool)
{
    for arg in proc_def.args
    {
        if arg.name == name {
            return arg.type.name, arg.attr, true
        }
    }

    for statement in proc_def.statements
    {
        if raw_data(statement.token.text) > raw_data(pos.text) {
            break
        }

        decl, ok := statement.derived_statement.(^Ast_Var_Decl)
        if ok && decl.name == name {
            return decl.type.name, nil, true
        }
    }

    return {}, {}, false
}

Type_Lookup :: struct
{
    is_primitive: bool,
    name: string,
    decl: ^Ast_Struct_Decl,  // If not primitive
    attr: Maybe(Ast_Attribute)
}

search_name :: proc(ast: Ast, proc_def: ^Ast_Proc_Decl, name: string, pos: Token) -> (res: Type_Lookup, ok: bool)
{
    type_name, attr, ok_s := search_type_name_of_name(proc_def, name, pos)
    if !ok_s
    {
        fmt.printfln("Error: Could not find variable '%v'.", name)
        return {}, false
    }

    switch type_name
    {
        case "float": return { true, type_name, nil, attr }, true
        case "uint":  return { true, type_name, nil, attr }, true
        case "vec2":  return { true, type_name, nil, attr }, true
        case "vec3":  return { true, type_name, nil, attr }, true
        case "vec4":  return { true, type_name, nil, attr }, true
    }

    for decl in ast.scope.decls
    {
        #partial switch d in decl.derived_decl
        {
            case ^Ast_Struct_Decl:
            {
                if decl.name == type_name {
                    return { false, type_name, d, nil }, true
                }
            }
        }
    }

    return {}, false
}

type_to_glsl :: proc(type: ^Ast_Type) -> string
{
    to_concat: []string
    if type.is_ptr {
        to_concat = { "_res_ptr_", type.name }
    } else if type.is_slice {
        to_concat = { "_res_slice_", type.name }
    } else {
        to_concat = { type.name }
    }
    concatenated := strings.concatenate(to_concat)
    return concatenated
}

attribute_to_glsl :: proc(attribute: Ast_Attribute) -> string
{
    to_concat: []string
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
}

@(private="file")
writer: Writer

@(deferred_in = writer_scope_end)
writer_scope :: proc() -> bool
{
    writer_scope_begin()
    return true
}

writer_scope_begin :: proc()
{
    writer.indentation += 1
}

writer_scope_end :: proc()
{
    writer.indentation -= 1
}

write_preamble :: proc()
{
    writeln("#version 460")
    writeln("#extension GL_EXT_buffer_reference : require")
    writeln("#extension GL_EXT_buffer_reference2 : require")
    writeln("")
}

writefln :: proc(fmt_str: string, args: ..any)
{
    write_indentation()
    fmt.printfln(fmt_str, ..args)
}

writef :: proc(fmt_str: string, args: ..any)
{
    fmt.printf(fmt_str, ..args)
}

writeln :: proc(strings: ..any)
{
    write_indentation()
    fmt.println(..strings)
}

write_begin :: proc(strings: ..any)
{
    write_indentation()
    fmt.print(..strings)
}

write :: proc(strings: ..any)
{
    fmt.print(..strings)
}

write_indentation :: proc()
{
    for i in 0..<4*writer.indentation {
        fmt.print(" ")
    }
}
