use crate::string_cache::StringCache;
use crate::token::{phrase_to_string, PhraseGroup, Token};

use std::marker::PhantomData;

/// Represents a Throne rule.
///
/// A `Rule` is uniquely identified by its `id`.
/// Each input and output for a `Rule` is a [Phrase](crate::Phrase).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rule {
    pub id: i32,
    pub inputs: Vec<Vec<Token>>,
    pub outputs: Vec<Vec<Token>>,
    pub(crate) input_phrase_group_counts: Vec<usize>,
    pub source_span: LineColSpan,
    marker: PhantomData<()>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LineColSpan {
    pub line_start: usize,
    pub line_end: usize,
    pub col_start: usize,
    pub col_end: usize,
}

impl Rule {
    pub fn to_string(&self, string_cache: &StringCache) -> String {
        rule_to_string_common(
            self.id.to_string(),
            &self.inputs,
            &self.outputs,
            string_cache,
        )
    }
}

#[derive(Clone, Debug)]
pub struct RuleBuilder {
    pub inputs: Vec<Vec<Token>>,
    pub outputs: Vec<Vec<Token>>,
    input_phrase_group_counts: Option<Vec<usize>>,
    source_span: LineColSpan,
}

impl RuleBuilder {
    pub fn new(
        inputs: Vec<Vec<Token>>,
        outputs: Vec<Vec<Token>>,
        source_span: LineColSpan,
    ) -> Self {
        RuleBuilder {
            inputs,
            outputs,
            input_phrase_group_counts: None,
            source_span,
        }
    }

    pub(crate) fn input_phrase_group_counts(mut self, v: Vec<usize>) -> Self {
        self.input_phrase_group_counts = Some(v);
        self
    }

    pub fn build(mut self, id: i32) -> Rule {
        let input_phrase_group_counts = self
            .input_phrase_group_counts
            .take()
            .unwrap_or_else(|| self.inputs.iter().map(|p| p.groups().count()).collect());
        Rule {
            id,
            inputs: self.inputs,
            outputs: self.outputs,
            input_phrase_group_counts,
            source_span: self.source_span,
            marker: PhantomData,
        }
    }

    pub fn to_string(&self, string_cache: &StringCache) -> String {
        rule_to_string_common("?".to_string(), &self.inputs, &self.outputs, string_cache)
    }
}

fn rule_to_string_common(
    id_str: String,
    inputs: &[Vec<Token>],
    outputs: &[Vec<Token>],
    string_cache: &StringCache,
) -> String {
    let inputs = inputs
        .iter()
        .map(|p| phrase_to_string(p, string_cache))
        .collect::<Vec<_>>()
        .join(" . ");

    let outputs = outputs
        .iter()
        .map(|p| phrase_to_string(p, string_cache))
        .collect::<Vec<_>>()
        .join(" . ");

    format!("{:5}: {} = {}", id_str, inputs, outputs)
}
