use crate::string_cache::StringCache;
use crate::token::{build_phrase, Token};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rule {
    pub id: i32,
    pub inputs: Vec<Vec<Token>>,
    pub outputs: Vec<Vec<Token>>,
}

impl Rule {
    pub fn new(id: i32, inputs: Vec<Vec<Token>>, outputs: Vec<Vec<Token>>) -> Rule {
        Rule {
            id,
            inputs,
            outputs,
        }
    }

    pub fn to_string(&self, string_cache: &StringCache) -> String {
        let inputs = self
            .inputs
            .iter()
            .map(|p| build_phrase(p, string_cache))
            .collect::<Vec<_>>()
            .join(" . ");

        let outputs = self
            .outputs
            .iter()
            .map(|p| build_phrase(p, string_cache))
            .collect::<Vec<_>>()
            .join(" . ");

        format!("{:5}: {} = {}", self.id, inputs, outputs)
    }
}
