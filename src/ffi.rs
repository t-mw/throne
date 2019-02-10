use std::ffi::CStr;
use std::mem::transmute;
use std::os::raw::{c_char, c_void};
use std::slice;

use ceptre::{Atom, Context, Token};

#[repr(C)]
pub struct CRule {
    id: i32,
}

#[no_mangle]
pub extern "C" fn ceptre_context_create_from_text(string_ptr: *const c_char) -> *mut Context {
    let cstr = unsafe { CStr::from_ptr(string_ptr) };
    unsafe { transmute(Box::new(Context::from_text(cstr.to_str().unwrap()))) }
}

#[no_mangle]
pub extern "C" fn ceptre_context_destroy(context: *mut Context) {
    let _drop: Box<Context> = unsafe { transmute(context) };
}

#[no_mangle]
pub extern "C" fn ceptre_update(context: *mut Context) {
    let context = unsafe { &mut *context };
    context.update(|_: &[Token]| None);
}

#[no_mangle]
pub extern "C" fn ceptre_context_string_to_atom(
    context: *mut Context,
    string_ptr: *const c_char,
) -> Atom {
    let context = unsafe { &mut *context };
    let cstr = unsafe { CStr::from_ptr(string_ptr) };
    context.str_to_atom(cstr.to_str().unwrap())
}

#[no_mangle]
pub extern "C" fn ceptre_context_find_matching_rules(
    context: *mut Context,
    side_input: extern "C" fn(p: *const Atom, p_len: usize, data: *mut c_void) -> bool,
    side_input_data: *mut c_void,
    result_ptr: *mut CRule,
    result_len: usize,
) -> usize {
    let context = unsafe { &mut *context };
    let result = unsafe { slice::from_raw_parts_mut(result_ptr, result_len) };

    let mut side_input_p = vec![];

    let rules = context.find_matching_rules(|p: &[Token]| {
        side_input_p.clear();
        side_input_p.extend(p.iter().map(|t| t.string));

        if side_input(side_input_p.as_ptr(), side_input_p.len(), side_input_data) {
            Some(vec![])
        } else {
            None
        }
    });

    let len = rules.len().min(result_len);

    for i in 0..len {
        result[i] = CRule { id: rules[i].id };
    }

    return len;
}

#[no_mangle]
pub extern "C" fn ceptre_context_find_phrase5(
    context: *mut Context,
    atom_ptr1: *const Atom,
    atom_ptr2: *const Atom,
    atom_ptr3: *const Atom,
    atom_ptr4: *const Atom,
    atom_ptr5: *const Atom,
    result_ptr: *mut Atom,
    result_len: usize,
) -> usize {
    let context = unsafe { &mut *context };
    let result = unsafe { slice::from_raw_parts_mut(result_ptr, result_len) };

    let ptr_to_atom = |atom_ptr: *const Atom| unsafe { atom_ptr.as_ref() };

    if let Some(phrase) = context.find_phrase5(
        ptr_to_atom(atom_ptr1),
        ptr_to_atom(atom_ptr2),
        ptr_to_atom(atom_ptr3),
        ptr_to_atom(atom_ptr4),
        ptr_to_atom(atom_ptr5),
    ) {
        let len = phrase.len().min(result_len);

        for i in 0..len {
            result[i] = phrase[i].string;
        }

        len
    } else {
        0
    }
}
