
package main

import "base:runtime"

typecheck_ast :: proc(ast: Ast, input_path: string, allocator: runtime.Allocator) -> bool
{
    context.allocator = allocator

    c := Checker {
        ast = ast,
        input_path = input_path,
        scope = ast.scope,
        error = false,
        cur_proc = nil,
    }

    add_intrinsics()

    for decl in ast.scope.decls
    {
        switch decl.type.kind
        {
            case .Poison: {}
            case .Unknown: {}
            case .Proc:
            {
                for arg in decl.type.args
                {
                    resolve_type(&c, arg.type)
                }

                resolve_type(&c, decl.type.ret)
            }
            case .Struct:
            {
                for member in decl.type.members
                {
                    resolve_type(&c, member.type)
                }
            }
            case .Label: {}
            case .Primitive: {}
            case .Pointer: {}
            case .Slice: {}
        }
    }

    for proc_def in ast.procs
    {
        c.cur_proc = proc_def

        for decl in proc_def.scope.decls
        {
            resolve_type(&c, decl.type)

            if decl.attr != nil && decl.attr.?.type == .Data
            {
                if decl.type.kind != .Pointer && decl.type.kind != .Slice {
                    typecheck_error(&c, decl.token, "Variable declared with '@data' attribute must be of pointer or slice type.")
                }
            }
            if decl.attr != nil && decl.attr.?.type == .Indirect_Data
            {
                if decl.type.kind != .Pointer && decl.type.kind != .Slice {
                    typecheck_error(&c, decl.token, "Variable declared with '@indirect_data' attribute must be of pointer or slice type.")
                }
            }

            for decl_2 in proc_def.scope.decls
            {
                if decl_2.name == decl.name && raw_data(decl_2.token.text) < raw_data(decl.token.text)
                {
                    typecheck_error_redeclaration(&c, decl_2, decl)
                    break
                }
            }
        }

        old_scope := c.scope
        c.scope = proc_def.scope
        defer c.scope = old_scope

        for stmt in proc_def.statements {
            typecheck_statement(&c, stmt)
        }
    }

    return !c.error
}

Checker :: struct #all_or_none
{
    ast: Ast,
    cur_proc: ^Ast_Proc_Def,
    scope: ^Ast_Scope,
    input_path: string,
    error: bool,
}

typecheck_statement :: proc(using c: ^Checker, statement: ^Ast_Statement)
{
    switch stmt in statement.derived_statement
    {
        case ^Ast_Stmt_Expr:
        {
            typecheck_expr(c, stmt.expr)
        }
        case ^Ast_Assign:
        {
            typecheck_expr(c, stmt.lhs)
            typecheck_expr(c, stmt.rhs)

            if !same_type(stmt.lhs.type, stmt.rhs.type) {
                typecheck_error_mismatching_types(c, stmt.token, stmt.lhs.type, stmt.rhs.type)
            }
        }
        case ^Ast_Define_Var:
        {
            typecheck_expr(c, stmt.expr)
            if stmt.decl.type.kind == .Unknown
            {
                stmt.decl.type = stmt.expr.type
            }
            else
            {
                if !same_type(stmt.decl.type, stmt.expr.type) {
                    typecheck_error_mismatching_types(c, stmt.token, stmt.decl.type, stmt.expr.type)
                }
            }
        }
        case ^Ast_If:
        {
            typecheck_expr(c, stmt.cond)
            if stmt.cond.type.primitive_kind != .Bool {
                typecheck_error_mismatching_types(c, stmt.token, stmt.cond.type, &BOOL_TYPE)
            }

            // Then
            {
                old_scope := scope
                scope = stmt.scope
                defer scope = old_scope

                resolve_scope_decls(c)
                typecheck_statement_list(c, stmt.statements)
            }
            // Else
            if stmt.else_is_present
            {
                old_scope := scope
                scope = stmt.scope
                defer scope = old_scope
                resolve_scope_decls(c)

                if stmt.else_is_single {
                    typecheck_statement(c, stmt.else_single)
                } else {
                    typecheck_statement_list(c, stmt.else_multi_statements)
                }
            }
        }
        case ^Ast_For:
        {
            old_scope := scope
            scope = stmt.scope
            defer scope = old_scope

            if stmt.define != nil do typecheck_statement(c, stmt.define)
            if stmt.cond != nil   do typecheck_expr(c, stmt.cond)
            if stmt.iter != nil   do typecheck_statement(c, stmt.iter)
            typecheck_statement_list(c, stmt.statements)
        }
        case ^Ast_Break:
        {
        }
        case ^Ast_Continue:
        {
        }
        case ^Ast_Return:
        {
            typecheck_expr(c, stmt.expr)
            if !same_type(stmt.expr.type, cur_proc.decl.type.ret) {
                typecheck_error_mismatching_types(c, stmt.token, stmt.expr.type, cur_proc.decl.type.ret)
            }
        }
    }
}

typecheck_statement_list :: proc(using c: ^Checker, stmts: []^Ast_Statement)
{
    for stmt in stmts {
        typecheck_statement(c, stmt)
    }
}

typecheck_expr :: proc(using c: ^Checker, expression: ^Ast_Expr)
{
    expression.type = &POISON_TYPE

    switch expr in expression.derived_expr
    {
        case ^Ast_Binary_Expr:
        {
            typecheck_expr(c, expr.lhs)
            typecheck_expr(c, expr.rhs)

            expr.type = bin_op_result_type(expr.op, expr.lhs.type, expr.rhs.type)
            if expr.type == &POISON_TYPE {
                typecheck_error_mismatching_types(c, expr.token, expr.lhs.type, expr.rhs.type)
            }
        }
        case ^Ast_Ident_Expr:
        {
            decl := decl_lookup(c, expr.token)
            if decl == nil {
                typecheck_error(c, expr.token, "Undeclared identifier '%v'.", expr.token.text)
            } else {
                expr.type = decl.type
            }
        }
        case ^Ast_Lit_Expr:
        {
            switch v in expr.token.value
            {
                case u64: expr.type = &INT_TYPE
                case f32: expr.type = &FLOAT_TYPE
            }
        }
        case ^Ast_Member_Access:
        {
            typecheck_expr(c, expr.target)

            if expr.member_name == "xyz"
            {
                expr.type = &VEC3_TYPE
                break
            }
            else if expr.member_name == "xy"
            {
                expr.type = &VEC2_TYPE
                break
            }
            else if expr.member_name == "y"
            {
                expr.type = &FLOAT_TYPE
                break
            }
            else if expr.member_name == "w"
            {
                expr.type = &FLOAT_TYPE
                break
            }

            base := type_get_base(expr.target.type)
            if base.kind != .Struct {
                typecheck_error(c, expr.token, "Can't access members on this type.")
            }

            field_type := &POISON_TYPE
            for field in base.members
            {
                if field.name == expr.member_name
                {
                    field_type = field.type
                    break
                }
            }

            if field_type == &POISON_TYPE {
                typecheck_error(c, expr.token, "Member not found.")
            }

            expr.type = field_type
        }
        case ^Ast_Array_Access:
        {
            typecheck_expr(c, expr.target)
            typecheck_expr(c, expr.idx_expr)

            if expr.target.type.kind != .Slice {
                typecheck_error(c, expr.token, "Can't access array element of this type, it must be a slice.")
                expr.target.type = &POISON_TYPE
            }

            expr.type = expr.target.type.base
        }
        case ^Ast_Call:
        {
            for arg in expr.args {
                typecheck_expr(c, arg)
            }

            // Handle intrinsics
            target, is_ident := expr.target.derived_expr.(^Ast_Ident_Expr)
            if is_ident
            {
                num_floats: u32
                for arg in expr.args
                {
                    if arg.type.kind != .Primitive
                    {
                        num_floats = 0
                        break
                    }

                    if arg.type.primitive_kind == .Float {
                        num_floats += 1
                    } else if arg.type.primitive_kind == .Vec2 {
                        num_floats += 2
                    } else if arg.type.primitive_kind == .Vec3 {
                        num_floats += 3
                    } else if arg.type.primitive_kind == .Vec4 {
                        num_floats += 4
                    } else {
                        num_floats = 0
                        break
                    }
                }

                name := target.token.text
                if name == "vec2"
                {
                    if num_floats != 2 do typecheck_error(c, expr.token, "Incorrect constructor arguments.")
                    expr.type = &VEC2_TYPE
                    break
                }
                else if name == "vec3"
                {
                    if num_floats != 3 do typecheck_error(c, expr.token, "Incorrect constructor arguments.")
                    expr.type = &VEC3_TYPE
                    break
                }
                else if name == "vec4"
                {
                    if num_floats != 4 do typecheck_error(c, expr.token, "Incorrect constructor arguments.")
                    expr.type = &VEC4_TYPE
                    break
                }
            }

            // Regular procedure calls

            typecheck_expr(c, expr.target)
            if expr.target.type.kind != .Proc {
                typecheck_error(c, expr.token, "Can't call this type, must be a procedure.")
            }

            if len(expr.target.type.args) != len(expr.args) {
                typecheck_error(c, expr.token, "Incorrect number of arguments, expecting '%v', got '%v'.", len(expr.target.type.args), len(expr.args))
                break
            }

            for arg, i in expr.args
            {
                proc_decl_arg_type := expr.target.type.args[i].type

                typecheck_expr(c, arg)

                if !same_type(arg.type, proc_decl_arg_type) {
                    typecheck_error_mismatching_types(c, expr.token, arg.type, proc_decl_arg_type)
                }
            }

            expr.type = expr.target.type.ret
        }
    }
}

POISON_TYPE := Ast_Type { kind = .Poison }
FLOAT_TYPE := Ast_Type { kind = .Primitive, primitive_kind = .Float, name = { text = "float", line = {}, value = {}, type = {}, col_start = {} } }
UINT_TYPE := Ast_Type { kind = .Primitive, primitive_kind = .Uint, name = { text = "uint", line = {}, value = {}, type = {}, col_start = {} } }
INT_TYPE := Ast_Type { kind = .Primitive, primitive_kind = .Int, name = { text = "int", line = {}, value = {}, type = {}, col_start = {} } }
VEC2_TYPE := Ast_Type { kind = .Primitive, primitive_kind = .Vec2, name = { text = "vec2", line = 0, value = {}, type = {}, col_start = {} } }
VEC3_TYPE := Ast_Type { kind = .Primitive, primitive_kind = .Vec3, name = { text = "vec3", line = 0, value = {}, type = {}, col_start = {} } }
VEC4_TYPE := Ast_Type { kind = .Primitive, primitive_kind = .Vec4, name = { text = "vec4", line = 0, value = {}, type = {}, col_start = {} } }
BOOL_TYPE := Ast_Type { kind = .Primitive, primitive_kind = .Bool, name = { text = "bool", line = 0, value = {}, type = {}, col_start = {} } }
TEXTUREID_TYPE := Ast_Type { kind = .Primitive, primitive_kind = .Texture_ID, name = { text = "textureid", line = {}, value = {}, type = {}, col_start = {} } }
SAMPLERID_TYPE := Ast_Type { kind = .Primitive, primitive_kind = .Sampler_ID, name = { text = "samplerid", line = {}, value = {}, type = {}, col_start = {} } }
MAT4_TYPE := Ast_Type { kind = .Primitive, primitive_kind = .Mat4, name = { text = "mat4", line = 0, value = {}, type = {}, col_start = {} } }

same_type :: proc(type1: ^Ast_Type, type2: ^Ast_Type) -> bool
{
    if type1.kind == .Poison || type2.kind == .Poison do return false
    if type1 == nil || type2 == nil do return false
    if type1.kind != type2.kind do return false
    if type1.primitive_kind != type2.primitive_kind do return false
    if type1.name.text != type2.name.text do return false

    has_base := type1.kind != .Primitive && type1.kind != .Label
    if has_base && !same_type(type1.base, type2.base) do return false
    return true
}

type_get_base :: proc(type: ^Ast_Type) -> ^Ast_Type
{
    if type.kind == .Poison do return &POISON_TYPE
    if type.base == nil do return type
    return type_get_base(type.base)
}

decl_lookup :: proc(using c: ^Checker, token: Token) -> ^Ast_Decl
{
    cur_scope := scope
    for cur_scope != nil
    {
        for decl in cur_scope.decls
        {
            ignore_order := decl.type.kind == .Struct || decl.type.kind == .Proc
            if !ignore_order && raw_data(decl.token.text) > raw_data(token.text) {
                continue
            }
            if decl.name == token.text do return decl
        }

        cur_scope = cur_scope.enclosing_scope
    }

    for intr in INTRINSICS
    {
        ignore_order := intr.type.kind == .Struct || intr.type.kind == .Proc
        if !ignore_order && raw_data(intr.token.text) > raw_data(token.text) {
            continue
        }
        if intr.name == token.text do return intr
    }

    return nil
}

resolve_type :: proc(using c: ^Checker, type: ^Ast_Type)
{
    base := type_get_base(type)
    if base.kind == .Label
    {
        type_decl := decl_lookup(c, base.name)
        if type_decl == nil {
            typecheck_error(c, base.name, "Undeclared identifier '%v'.", base.name.text)
        } else {
            base.base = type_decl.type
        }
    }
}

typecheck_error :: proc(using c: ^Checker, token: Token, fmt_str: string, args: ..any)
{
    if error do return

    error_msg(input_path, token, fmt_str, ..args)
    error = true
}

typecheck_error_mismatching_types :: proc(using c: ^Checker, token: Token, type1: ^Ast_Type, type2: ^Ast_Type)
{
    if error do return

    scratch, _ := acquire_scratch()
    type1_str := type_to_string(type1, arena = scratch)
    type2_str := type_to_string(type2, arena = scratch)
    error_msg(input_path, token, "Incompatible types: '%v' and '%v'", type1_str, type2_str)
    error = true
}

typecheck_error_redeclaration :: proc(using c: ^Checker, decl_before: ^Ast_Decl, decl_after: ^Ast_Decl)
{
    if error do return

    error_msg(input_path, decl_after.token, "Redeclaration of '%v' in this scope.", decl_after.name)
    error = true
}

INTRINSICS: [dynamic]^Ast_Decl

add_intrinsics :: proc()
{
    add_intrinsic("sample", { &TEXTUREID_TYPE, &SAMPLERID_TYPE, &VEC2_TYPE }, { "tex_idx", "sampler_idx", "uv" }, &VEC4_TYPE)
    add_intrinsic("mix", { &VEC4_TYPE, &VEC4_TYPE, &FLOAT_TYPE }, { "a", "b", "t" }, &VEC4_TYPE)
    add_intrinsic("normalize", { &VEC3_TYPE }, { "v" }, &VEC3_TYPE)
}

add_intrinsic :: proc(name: string, args: []^Ast_Type, names: []string, ret: ^Ast_Type = nil)
{
    assert(len(args) == len(names))

    arg_decls := make([]^Ast_Decl, len(args))
    for &arg, i in arg_decls
    {
        arg = new(Ast_Decl)
        arg.type = args[i]
        arg.name = names[i]
    }

    decl := new(Ast_Decl)
    decl.name = name
    decl.type = new(Ast_Type)
    decl.type.kind = .Proc
    decl.type.args = arg_decls
    decl.type.ret = ret
    append(&INTRINSICS, decl)
}

// Returns &POISON_TYPE if the two types are not allowed
bin_op_result_type :: proc(op: Ast_Binary_Op, type1: ^Ast_Type, type2: ^Ast_Type) -> ^Ast_Type
{
    if op == .Mul && type1.primitive_kind == .Mat4
    {
        if type2.primitive_kind == .Vec4 do return &VEC4_TYPE
    }
    else if op == .Mul && type1.primitive_kind == .Vec4
    {
        if type2.primitive_kind == .Mat4 do return &VEC4_TYPE
    }

    is_compare := op == .Greater ||
                  op == .Less ||
                  op == .LE ||
                  op == .GE ||
                  op == .EQ ||
                  op == .NEQ
    if is_compare
    {
        if same_type(type1, type2) && is_compare do return &BOOL_TYPE
        else do return &POISON_TYPE
    }

    type_less := type1
    type_greater := type2
    if type_less.primitive_kind > type_greater.primitive_kind {
        type_less, type_greater = type_greater, type_less
    }

    if type_less.primitive_kind == .Float && type_greater.primitive_kind == .Vec2 {
        return type_greater
    }
    if type_less.primitive_kind == .Float && type_greater.primitive_kind == .Vec3 {
        return type_greater
    }
    if type_less.primitive_kind == .Float && type_greater.primitive_kind == .Vec4 {
        return type_greater
    }

    if same_type(type1, type2) do return type1
    return &POISON_TYPE
}

resolve_scope_decls :: proc(using c: ^Checker)
{
    for decl in scope.decls
    {
        resolve_type(c, decl.type)

        for decl_2 in scope.decls
        {
            if decl_2.name == decl.name && raw_data(decl_2.token.text) < raw_data(decl.token.text)
            {
                typecheck_error_redeclaration(c, decl_2, decl)
                break
            }
        }
    }
}
