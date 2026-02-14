use crate::context::Context;
use crate::string_cache::Atom;
use crate::token::Phrase;

use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::slice;

#[repr(C)]
pub struct CRule {
    id: i32,
}

/// # Safety
///
/// `string_ptr` must be non-null and point to a valid, NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn throne_context_create_from_text(
    string_ptr: *const c_char,
) -> *mut Context {
    let cstr = unsafe { CStr::from_ptr(string_ptr) };
    Box::into_raw(Box::new(
        Context::from_text(cstr.to_str().unwrap()).unwrap(),
    ))
}

/// # Safety
///
/// `context` must be non-null and must have been returned by
/// [throne_context_create_from_text]. It must not be used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn throne_context_destroy(context: *mut Context) {
    unsafe { drop(Box::from_raw(context)) };
}

/// # Safety
///
/// `context` must be non-null and point to a valid [Context].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn throne_update(context: *mut Context) {
    let context = unsafe { &mut *context };
    context.update().unwrap();
}

/// # Safety
///
/// `context` must be non-null and point to a valid [Context].
/// `string_ptr` must be non-null and point to a valid, NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn throne_context_string_to_atom(
    context: *mut Context,
    string_ptr: *const c_char,
) -> Atom {
    let context = unsafe { &mut *context };
    let cstr = unsafe { CStr::from_ptr(string_ptr) };
    context.str_to_atom(cstr.to_str().unwrap())
}

/// # Safety
///
/// `context` must be non-null and point to a valid [Context].
/// `result_ptr` must point to writable memory for at least `result_len` [CRule] values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn throne_context_find_matching_rules(
    context: *mut Context,
    side_input: extern "C" fn(p: *const Atom, p_len: usize, data: *mut c_void) -> bool,
    side_input_data: *mut c_void,
    result_ptr: *mut CRule,
    result_len: usize,
) -> usize {
    let context = unsafe { &mut *context };
    let result = unsafe { slice::from_raw_parts_mut(result_ptr, result_len) };

    let mut side_input_p = vec![];

    let rules = context
        .find_matching_rules(|p: &Phrase| {
            side_input_p.clear();
            side_input_p.extend(p.iter().map(|t| t.atom));

            if side_input(side_input_p.as_ptr(), side_input_p.len(), side_input_data) {
                Some(vec![])
            } else {
                None
            }
        })
        .unwrap();

    let len = rules.len().min(result_len);

    for i in 0..len {
        result[i] = CRule { id: rules[i].id };
    }

    len
}
