
#+feature dynamic-literals

package main

import "core:fmt"
import "base:runtime"
import "core:slice"

Any_Node :: union
{
    ^Ast_Decl_Arg,
    Any_Statement,
    Any_Expr,
    Any_Decl,
}

Ast :: struct
{
    used_types: map[Ast_Type]struct{},
    used_in_locations: map[u32]Ast_Type,
    used_out_locations: map[u32]Ast_Type,
    used_data_type: string,
    scope: ^Ast_Scope,
}

Ast_Node :: struct
{
    token: Token,
    derived: Any_Node,
}

Ast_Scope :: struct
{
    enclosing_scope: ^Ast_Scope,  // Can be nil
    decls: []^Ast_Decl,
}

// Declarations (global)

Any_Decl :: union
{
    ^Ast_Struct_Decl,
    ^Ast_Proc_Decl,
}

Ast_Decl :: struct
{
    using base: Ast_Node,
    derived_decl: Any_Decl,
    name: string,
}

Ast_Decl_Arg :: struct
{
    using base: Ast_Node,
    name: string,
    type: ^Ast_Type,
    attr: Maybe(Ast_Attribute)
}

Ast_Struct_Decl :: struct
{
    using base_decl: Ast_Decl,
    fields: []^Ast_Decl_Arg,
}

Ast_Proc_Decl :: struct
{
    using base_decl: Ast_Decl,
    args: []^Ast_Decl_Arg,
    return_type: ^Ast_Type,
    return_attr: Maybe(Ast_Attribute),
    statements: []^Ast_Statement,
}

// Expressions

Any_Expr :: union
{
    ^Ast_Binary_Expr,
    ^Ast_Member_Access,
    ^Ast_Array_Access,
    ^Ast_Ident_Expr,
    ^Ast_Lit_Expr,
    ^Ast_Call,
}

Ast_Expr :: struct
{
    using base: Ast_Node,
    derived_expr: Any_Expr,
    type: ^Ast_Type,
}

Ast_Attribute_Type :: enum
{
    Vert_ID,
    Position,
    Data,

    // With args:
    Out_Loc,
    In_Loc,
}

Ast_Attribute :: struct
{
    type: Ast_Attribute_Type,
    arg: u32
}

Ast_Binary_Op :: enum
{
    Add,
    Minus,
    Mul,
    Div,
}

Ast_Binary_Expr :: struct
{
    using base_expr: Ast_Expr,
    lhs: ^Ast_Expr,
    rhs: ^Ast_Expr,
    op: Ast_Binary_Op,
}

Ast_Member_Access :: struct
{
    using base_expr: Ast_Expr,
    target: ^Ast_Expr,
    member_name: string,
}

Ast_Array_Access :: struct
{
    using base_expr: Ast_Expr,
    target: ^Ast_Expr,
    idx_expr: ^Ast_Expr,
}

Ast_Call :: struct
{
    using base_expr: Ast_Expr,
    target: ^Ast_Expr,
    args: []^Ast_Expr,
}

Ast_Ident_Expr :: struct
{
    using base_expr: Ast_Expr
}

Ast_Lit_Expr :: struct
{
    using base_expr: Ast_Expr
}

// Statements

Any_Statement :: union
{
    ^Ast_Assign,
    ^Ast_Var_Decl,
    ^Ast_Stmt_Expr,
    ^Ast_Return,
}

Ast_Statement :: struct
{
    using base: Ast_Node,
    derived_statement: Any_Statement,
}

Ast_Assign :: struct
{
    using base_statement: Ast_Statement,
    lhs: ^Ast_Expr,
    rhs: ^Ast_Expr
}

Ast_Var_Decl :: struct
{
    using base_statement: Ast_Statement,
    name: string,
    type: ^Ast_Type,
}

Ast_Stmt_Expr :: struct
{
    using base_statement: Ast_Statement,
    expr: ^Ast_Expr,
}

Ast_Return :: struct
{
    using base_statement: Ast_Statement,
    expr: ^Ast_Expr,
}

// Types

Ast_Type :: struct
{
    is_ptr: bool,
    is_slice: bool,
    name: string,
    struct_decl: ^Ast_Struct_Decl,  // Can be nil
    attr: Maybe(Ast_Attribute)
}

parse_file :: proc(filename: string, tokens: []Token, allocator: runtime.Allocator) -> (Ast, bool)
{
    context.allocator = allocator

    parser := Parser {
        tokens = tokens,
        filename = filename,
    }
    ast := _parse_file(&parser)
    return ast, !parser.error
}

Parser :: struct
{
    tokens: []Token,
    filename: string,
    at: u32,
    error: bool,
    cur_scope: ^Ast_Scope,
    used_types: map[Ast_Type]struct{},
    used_out_locations: map[u32]Ast_Type,
    used_in_locations: map[u32]Ast_Type,
    used_data_type: string,
}

_parse_file :: proc(using p: ^Parser) -> Ast
{
    ast := Ast {
        scope = new(Ast_Scope)
    }

    cur_scope = ast.scope
    scratch, _ := acquire_scratch()
    tmp_list := make([dynamic]^Ast_Decl, allocator = scratch)

    loop: for true
    {
        #partial switch tokens[at].type
        {
            case .Ident:
            {
                if tokens[at+1].type == .Colon &&
                   tokens[at+2].type == .Colon &&
                   tokens[at+3].type == .Struct
                {
                    append(&tmp_list, parse_struct_def(p))
                }
                else if tokens[at+1].type == .Colon &&
                        tokens[at+2].type == .Colon &&
                        tokens[at+3].type == .LParen
                {
                    append(&tmp_list, parse_proc_def(p))
                }
                else
                {
                    parse_error(p, "Expecting ':: (' or ':: struct' after the identifier")
                    break loop
                }
            }
            case .EOS: break loop
            case:
            {
                parse_error(p, "Expecting an identifier at top level.")
                break loop
            }
        }
    }

    ast.scope.decls = slice.clone(tmp_list[:])
    ast.used_types = used_types
    ast.used_out_locations = used_out_locations
    ast.used_in_locations = used_in_locations
    ast.used_data_type = used_data_type
    return ast
}

parse_struct_def :: proc(using p: ^Parser) -> ^Ast_Decl
{
    node := make_decl(p, Ast_Struct_Decl)

    ident := required_token(p, .Ident)
    node.name = ident.text

    required_token(p, .Colon)
    required_token(p, .Colon)
    required_token(p, .Struct)
    required_token(p, .LBrace)
    node.fields = parse_decl_list(p)
    required_token(p, .RBrace)
    return node
}

parse_proc_def :: proc(using p: ^Parser) -> ^Ast_Decl
{
    node := make_decl(p, Ast_Proc_Decl)

    ident := required_token(p, .Ident)
    node.name = ident.text

    required_token(p, .Colon)
    required_token(p, .Colon)
    required_token(p, .LParen)
    node.args = parse_decl_list(p)
    required_token(p, .RParen)

    if optional_token(p, .Arrow)
    {
        node.return_type = parse_type(p)
        if tokens[at].type == .Attribute {
            node.return_attr = parse_attribute(p)
            if node.return_attr != nil
            {
                if node.return_attr.?.type == .In_Loc {
                    used_in_locations[node.return_attr.?.arg] = node.return_type^
                } else if node.return_attr.?.type == .Out_Loc {
                    used_out_locations[node.return_attr.?.arg] = node.return_type^
                }
            }
        }
    }

    required_token(p, .LBrace)
    node.statements = parse_statement_list(p)
    required_token(p, .RBrace)
    return node
}

parse_statement_list :: proc(using p: ^Parser) -> []^Ast_Statement
{
    scratch, _ := acquire_scratch()
    tmp_list := make([dynamic]^Ast_Statement, allocator = scratch)
    for true
    {
        append(&tmp_list, parse_statement(p))
        if tokens[at].type == .RBrace || tokens[at].type == .EOS do break
        if error do break
    }
    return slice.clone(tmp_list[:])
}

parse_statement :: proc(using p: ^Parser) -> ^Ast_Statement
{
    for optional_token(p, .Semi) {}

    cursor := at
    for ; tokens[cursor].type != .Semi && tokens[cursor].type != .Assign && tokens[cursor].type != .EOS; cursor += 1 { }

    found_assign := tokens[cursor].type == .Assign
    node: ^Ast_Statement
    if found_assign
    {
        node = parse_assign(p)
    }
    else if optional_token(p, .Return)
    {
        ret_stmt := make_statement(p, Ast_Return)
        ret_stmt.expr = parse_expr(p)
        node = ret_stmt
    }
    else if tokens[at].type == .Ident && tokens[at+1].type == .Colon
    {
        node = parse_var_decl(p)
    }
    else
    {
        stmt_expr := make_statement(p, Ast_Stmt_Expr)
        stmt_expr.expr = parse_expr(p)
        node = stmt_expr
    }

    required_token(p, .Semi)
    return node
}

parse_assign :: proc(using p: ^Parser) -> ^Ast_Statement
{
    node := make_statement(p, Ast_Assign)
    node.lhs = parse_expr(p)
    required_token(p, .Assign)
    node.rhs = parse_expr(p)
    return node
}

parse_var_decl :: proc(using p: ^Parser) -> ^Ast_Statement
{
    node := make_statement(p, Ast_Var_Decl)

    ident := required_token(p, .Ident)
    node.name = ident.text

    required_token(p, .Colon)
    node.type = parse_type(p)
    return node
}

parse_expr :: proc(using p: ^Parser, prec: int = max(int)) -> ^Ast_Expr
{
    lhs: ^Ast_Expr

    // Prefix operators

    // Postfix operators
    lhs = parse_postfix_expr(p)

    // Binary operators
    for true
    {
        op, found := Op_Precedence[tokens[at].type]
        undo_recurse := false
        undo_recurse |= !found
        undo_recurse |= op.prec > prec  // If it's less important (=greater priority) don't recurse.
        if undo_recurse do return lhs

        // Recurse
        bin_op := make_node(p, Ast_Binary_Expr)
        bin_op.op = op.op
        bin_op.lhs = lhs
        bin_op.rhs = parse_expr(p)
        lhs = bin_op
    }

    return lhs
}

parse_primary_expr :: proc(using p: ^Parser) -> ^Ast_Expr
{
    expr: ^Ast_Expr
    if optional_token(p, .LParen)
    {
        internal := parse_expr(p)
        required_token(p, .RParen)
        expr = internal
    }
    else if token := tokens[at]; token.type == .Ident
    {
        expr = make_expr(p, Ast_Ident_Expr)
        at += 1
    }
    else if token := tokens[at]; token.type == .NumLit
    {
        expr = make_expr(p, Ast_Lit_Expr)
        at += 1
    }

    return expr
}

parse_postfix_expr :: proc(using p: ^Parser) -> ^Ast_Expr
{
    expr := parse_primary_expr(p)
    loop: for true
    {
        #partial switch tokens[at].type
        {
            case .Dot:
            {
                member_access := make_expr(p, Ast_Member_Access)
                at += 1

                ident := required_token(p, .Ident)
                member_access.member_name = ident.text
                member_access.target = expr
                expr = member_access
            }
            case .LParen:
            {
                call := make_expr(p, Ast_Call)
                call.target = expr
                at += 1

                if tokens[at].type != .RParen
                {
                    scratch, _ := acquire_scratch()
                    tmp_list := make([dynamic]^Ast_Expr, allocator = scratch)
                    for true
                    {
                        append(&tmp_list, parse_expr(p))
                        comma_present := optional_token(p, .Comma)
                        if !comma_present do break
                        if comma_present && (tokens[at].type == .RParen || tokens[at].type == .RBrace) do break
                        if error do break
                    }

                    call.args = slice.clone(tmp_list[:])
                }

                required_token(p, .RParen)

                expr = call
            }
            case .LBracket:
            {
                array_access := make_expr(p, Ast_Array_Access)
                at += 1

                array_access.idx_expr = parse_expr(p)
                array_access.target = expr
                expr = array_access

                required_token(p, .RBracket)
            }
            case: break loop
        }
    }

    return expr
}

parse_decl_list_elem :: proc(using p: ^Parser) -> ^Ast_Decl_Arg
{
    node := make_node(p, Ast_Decl_Arg)

    ident := required_token(p, .Ident)
    required_token(p, .Colon)

    node.name = ident.text

    node.type = parse_type(p)
    node.attr = parse_attribute(p)
    if node.attr != nil
    {
        if node.attr.?.type == .Data {
            used_data_type = node.type.name
        }

        if node.attr.?.type == .In_Loc {
            used_in_locations[node.attr.?.arg] = node.type^
        } else if node.attr.?.type == .Out_Loc {
            used_out_locations[node.attr.?.arg] = node.type^
        }
    }

    return node
}

parse_type :: proc(using p: ^Parser) -> ^Ast_Type
{
    node := new(Ast_Type)

    if optional_token(p, .LBracket)
    {
        required_token(p, .RBracket)
        node.is_slice = true
    }
    else if optional_token(p, .Caret)
    {
        node.is_ptr = true
    }

    ident := required_token(p, .Ident)
    node.name = ident.text

    // Add to the global type table
    used_types[node^] = {}
    return node
}

parse_decl_list :: proc(using p: ^Parser) -> []^Ast_Decl_Arg
{
    scratch, _ := acquire_scratch()
    tmp_list := make([dynamic]^Ast_Decl_Arg, allocator = scratch)
    for true
    {
        append(&tmp_list, parse_decl_list_elem(p))
        comma_present := optional_token(p, .Comma)
        if !comma_present do break
        if comma_present && (tokens[at].type == .RParen || tokens[at].type == .RBrace) do break
        if error do break
    }
    return slice.clone(tmp_list[:])
}

parse_attribute :: proc(using p: ^Parser) -> Maybe(Ast_Attribute)
{
    if tokens[at].type != .Attribute do return nil

    attr := Ast_Attribute {}

    token := required_token(p, .Attribute)

    switch token.text
    {
        case "vert_id": attr.type = .Vert_ID
        case "position": attr.type = .Position
        case "data": attr.type = .Data
        case "in_loc":
        {
            // ??? Why is the compiler making me do this?
            attr.type, _ = .In_Loc,
            required_token(p, .LParen)
            num_token := required_token(p, .NumLit)
            val, ok := num_token.value.(u64)
            if !ok do parse_error_on_token(p, num_token, "Expecting integer value on attribute arguments.")
            attr.arg = u32(val)
            required_token(p, .RParen)
        }
        case "out_loc":
        {
            // ??? Why is the compiler making me do this?
            attr.type, _ = .Out_Loc,
            required_token(p, .LParen)
            num_token := required_token(p, .NumLit)
            val, ok := num_token.value.(u64)
            if !ok do parse_error_on_token(p, num_token, "Expecting integer value on attribute arguments.")
            attr.arg = u32(val)
            required_token(p, .RParen)
        }
        case:
        {
            parse_error(p, "Unknown attribute '%v'.", token.text)
        }
    }

    return attr
}

make_node :: proc(using p: ^Parser, $T: typeid) -> ^T
{
    node := new(T)
    node.token = tokens[at]
    return node
}

make_expr :: proc(using p: ^Parser, $T: typeid) -> ^T
{
    node := new(T)
    node.token = tokens[at]
    node.derived_expr = node
    return node
}

make_decl :: proc(using p: ^Parser, $T: typeid) -> ^T
{
    node := new(T)
    node.token = tokens[at]
    node.derived_decl = node
    return node
}

make_statement :: proc(using p: ^Parser, $T: typeid) -> ^T
{
    node := new(T)
    node.token = tokens[at]
    node.derived_statement = node
    return node
}

parse_error :: proc(using p: ^Parser, fmt_str: string, args: ..any)
{
    parse_error_on_token(p, tokens[at], fmt_str, ..args)
}

parse_error_on_token :: proc(using p: ^Parser, token: Token, fmt_str: string, args: ..any)
{
    if error do return

    error_msg(filename, token, fmt_str, ..args)
    error = true
}

required_token :: proc(using p: ^Parser, type: Token_Type) -> Token
{
    if tokens[at].type != type
    {
        parse_error(p, "Unexpected token '%v': expecting '%v'", tokens[at].text, type)
        return {}
    }

    at += 1
    return tokens[at-1]
}

optional_token :: proc(using p: ^Parser, type: Token_Type) -> bool
{
    if tokens[at].type == type
    {
        at += 1
        return true
    }

    return false
}

// Operator precedence
Op_Info :: struct
{
    prec: int,
    op: Ast_Binary_Op,
}
Op_Precedence := map[Token_Type]Op_Info {
    .Mul = { 3, .Mul },
    .Div = { 3, .Div },

    .Plus  = { 4, .Add },
    .Minus = { 4, .Minus },
}
