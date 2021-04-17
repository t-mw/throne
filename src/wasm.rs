use wasm_bindgen::prelude::*;

use crate::string_cache::{Atom, StringCache};
use crate::throne::{update, Context as ThroneContext};
use crate::token::{Phrase, PhraseGroup};

#[wasm_bindgen]
pub fn init() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub struct Context {
    throne_context: ThroneContext,
}

#[wasm_bindgen]
impl Context {
    pub fn from_text(text: &str) -> Self {
        Context {
            throne_context: ThroneContext::from_text(text),
        }
    }

    pub fn update(&mut self, side_input: Option<js_sys::Function>) {
        if let Some(side_input) = side_input {
            let core = &mut self.throne_context.core;
            let string_cache = &self.throne_context.string_cache;
            update(core, |phrase: &Phrase| {
                let js_phrase = js_value_from_phrase(phrase, string_cache);
                if side_input
                    .call1(&JsValue::null(), &js_phrase)
                    .ok()
                    .filter(|v| v.is_truthy())
                    .is_some()
                {
                    Some(phrase.to_vec())
                } else {
                    None
                }
            });
        } else {
            self.throne_context.update(|_: &Phrase| None);
        }
    }

    pub fn get_state(&self) -> JsValue {
        let string_cache = &self.throne_context.string_cache;
        let js_phrases = self
            .throne_context
            .core
            .state
            .get_all()
            .iter()
            .map(|phrase| js_value_from_phrase(phrase, string_cache))
            .collect::<js_sys::Array>();
        JsValue::from(js_phrases)
    }

    pub fn print(&self) {
        log(&format!("{}", self.throne_context));
    }
}

fn js_value_from_phrase(phrase: &Phrase, string_cache: &StringCache) -> JsValue {
    let mut result = vec![];
    let mut group_n = 0;

    // NB: optimizable, because get_group finds all earlier groups on each call
    while let Some(group) = phrase.get_group(group_n) {
        if group.len() == 1 {
            result.push(js_value_from_atom(group[0].atom, string_cache));
        } else {
            result.push(js_value_from_phrase(group, string_cache));
        }
        group_n += 1;
    }

    JsValue::from(result.iter().collect::<js_sys::Array>())
}

fn js_value_from_atom(atom: Atom, string_cache: &StringCache) -> JsValue {
    if let Some(string) = string_cache.atom_to_str(atom) {
        JsValue::from(string)
    } else if let Some(n) = StringCache::atom_to_number(atom) {
        JsValue::from(n)
    } else {
        JsValue::null()
    }
}

#[cfg(test)]
mod tests {
    extern crate wasm_bindgen_test;
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;
    use crate::token::tokenize;

    #[wasm_bindgen_test]
    fn test_js_value_from_phrase_nested() {
        let mut string_cache = StringCache::new();
        let phrase = tokenize("t1 (t21 (t221 t222 t223) t23) t3", &mut string_cache);
        let js_phrase = js_value_from_phrase(&phrase, &string_cache);
        assert_eq!(
            format!("{:?}", js_phrase),
            r#"JsValue(["t1", ["t21", ["t221", "t222", "t223"], "t23"], "t3"])"#
        );
    }
}
