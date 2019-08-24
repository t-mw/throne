use regex::Regex;

use crate::string_cache::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenFlag {
    None,
    Variable,
    Side,
    BackwardsPred(BackwardsPred),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackwardsPred {
    Plus,
    Lt,
    Gt,
    Lte,
    Gte,
    ModNeg,
    Custom,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Token {
    pub string: Atom,
    pub is_negated: bool,
    pub is_consuming: bool,
    pub flag: TokenFlag,
    pub open_depth: u8,
    pub close_depth: u8,
}

impl Token {
    pub fn new(
        string: &str,
        open_depth: u8,
        close_depth: u8,
        string_cache: &mut StringCache,
    ) -> Token {
        let mut string = string;

        let is_consuming = if let Some('?') = string.chars().next() {
            string = string.get(1..).expect("get");
            false
        } else {
            true
        };

        let mut is_negated = false;
        let mut is_side = false;
        match string.chars().next() {
            Some('!') => {
                is_negated = true;
                string = string.get(1..).expect("get");
            }
            Some('^') => {
                is_side = true;
                string = string.get(1..).expect("get");
            }
            _ => {}
        }

        let mut chars = string.chars();
        let first_char = chars.next();
        let is_var = first_char.map(|c| c.is_ascii_uppercase()).unwrap_or(false)
            && chars.all(|c| c.is_numeric() || !c.is_ascii_lowercase());

        let second_char = string.chars().nth(1);

        let backwards_pred = match (first_char.unwrap_or(' '), second_char.unwrap_or(' ')) {
            ('<', '<') => Some(BackwardsPred::Custom),
            ('%', '%') => Some(BackwardsPred::ModNeg),
            ('<', '=') => Some(BackwardsPred::Lte),
            ('>', '=') => Some(BackwardsPred::Gte),
            ('<', _) => Some(BackwardsPred::Lt),
            ('>', _) => Some(BackwardsPred::Gt),
            ('+', _) => Some(BackwardsPred::Plus),

            _ => None,
        };

        let atom = string_cache.str_to_atom(string);

        let flag = match (is_var, is_side, backwards_pred) {
            (false, false, None) => TokenFlag::None,
            (true, false, None) => TokenFlag::Variable,
            (false, true, None) => TokenFlag::Side,
            (false, false, Some(v)) => TokenFlag::BackwardsPred(v),
            _ => unreachable!(),
        };

        Token {
            string: atom,
            flag,
            is_negated,
            is_consuming,
            open_depth,
            close_depth,
        }
    }

    pub fn new_atom(atom: Atom, open_depth: u8, close_depth: u8) -> Token {
        Token {
            string: atom,
            flag: TokenFlag::None,
            is_negated: false,
            is_consuming: true,
            open_depth,
            close_depth,
        }
    }

    pub fn new_number(n: i32, open_depth: u8, close_depth: u8) -> Token {
        Token {
            string: StringCache::number_to_atom(n),
            flag: TokenFlag::None,
            is_negated: false,
            is_consuming: true,
            open_depth,
            close_depth,
        }
    }

    pub fn as_str<'a>(&self, string_cache: &'a StringCache) -> Option<&'a str> {
        string_cache.atom_to_str(self.string)
    }

    pub fn as_number(&self) -> Option<i32> {
        StringCache::atom_to_number(self.string)
    }
}

pub fn tokenize(string: &str, string_cache: &mut StringCache) -> Vec<Token> {
    let mut string = format!("({})", string);

    lazy_static! {
        static ref RE1: Regex = Regex::new(r"\(\s*(\S+|`[^`]+`)\s*\)").unwrap();
    }

    loop {
        // remove instances of brackets surrounding single atoms
        let string1 = string.clone();
        let string2 = RE1.replace_all(&string1, "$1");

        if string1 == string2 {
            break;
        } else {
            string = string2.into_owned();
        }
    }

    lazy_static! {
        static ref RE2: Regex = Regex::new(r"`(.*?)`").unwrap();
    }

    let string1 = string.clone();
    let mut strings = RE2
        .captures_iter(&string1)
        .map(|c| c.get(1).expect("string_capture").as_str());

    string = RE2.replace_all(&string, "`").to_string();

    lazy_static! {
        static ref RE3: Regex = Regex::new(r"\(|\)|\s+|[^\(\)\s]+").unwrap();
    }

    let tokens = RE3
        .find_iter(&string)
        .map(|m| m.as_str())
        .filter(|s| !s.trim().is_empty())
        .collect::<Vec<_>>();

    let mut result = vec![];

    let mut open_depth = 0;
    let mut close_depth = 0;

    for (i, token) in tokens.iter().enumerate() {
        if *token == "(" {
            open_depth += 1;
            continue;
        }

        for t in tokens.iter().skip(i + 1) {
            if *t == ")" {
                close_depth += 1;
            } else {
                break;
            }
        }

        if *token == ")" {
            continue;
        }

        if *token == "`" {
            result.push(Token::new(
                strings.next().expect("string"),
                open_depth,
                close_depth,
                string_cache,
            ));
        } else {
            result.push(Token::new(token, open_depth, close_depth, string_cache));
        }
        open_depth = 0;
        close_depth = 0;
    }

    result
}

pub type Phrase = [Token];
pub type VecPhrase = Vec<Token>;

pub trait PhraseGroup {
    fn get_group(&self, n: usize) -> Option<&[Token]>;
    fn normalize(&self) -> Vec<Token>;
}

impl PhraseGroup for Phrase {
    fn get_group(&self, n: usize) -> Option<&[Token]> {
        let mut current_group = 0;
        let mut depth = 0;

        let mut start = None;

        for (i, t) in self.iter().enumerate() {
            let depth_before = depth;

            depth += t.open_depth;
            depth -= t.close_depth;

            if depth_before <= 1 {
                start = Some(i);
            }

            if depth <= 1 {
                if current_group == n {
                    return Some(&self[start.unwrap()..i + 1]);
                }

                current_group += 1;
            }
        }

        None
    }

    fn normalize(&self) -> Vec<Token> {
        let len = self.len();
        if len == 0 {
            return vec![];
        }

        let mut vec = self.to_vec();

        let mut interior_depth: i32 = 0;
        let mut min_interior_depth: i32 = 0;
        for (i, t) in vec.iter().enumerate() {
            if i == 0 {
                interior_depth -= t.close_depth as i32;
            } else if i == vec.len() - 1 {
                interior_depth += t.open_depth as i32;
            } else {
                interior_depth += t.open_depth as i32;
                interior_depth -= t.close_depth as i32;
            }

            min_interior_depth = interior_depth.min(min_interior_depth);
        }

        vec[0].open_depth = 1 + (-min_interior_depth) as u8;
        vec[len - 1].close_depth = 1 + (interior_depth - min_interior_depth) as u8;

        return vec;
    }
}

#[inline]
pub fn token_equal(
    a: &Token,
    b: &Token,
    ignore_depth: bool,
    a_depth_diffs: Option<(u8, u8)>,
    b_depth_diffs: Option<(u8, u8)>,
) -> bool {
    let a_depth_diffs = a_depth_diffs.unwrap_or((0, 0));
    let b_depth_diffs = b_depth_diffs.unwrap_or((0, 0));

    a.string == b.string
        && a.is_negated == b.is_negated
        && (ignore_depth
            || (a.open_depth as i32 - a_depth_diffs.0 as i32
                == b.open_depth as i32 - b_depth_diffs.0 as i32
                && a.close_depth as i32 - a_depth_diffs.1 as i32
                    == b.close_depth as i32 - b_depth_diffs.1 as i32))
}

pub fn is_var_token(token: &Token) -> bool {
    token.flag == TokenFlag::Variable
}

pub fn is_backwards_pred(tokens: &Phrase) -> bool {
    if let TokenFlag::BackwardsPred(_) = tokens[0].flag {
        true
    } else {
        false
    }
}

pub fn is_side_pred(tokens: &Phrase) -> bool {
    tokens[0].flag == TokenFlag::Side
}

pub fn is_negated_pred(tokens: &Phrase) -> bool {
    tokens[0].is_negated
}

pub fn is_concrete_pred(tokens: &Phrase) -> bool {
    !is_negated_pred(tokens) && tokens[0].flag == TokenFlag::None
}

pub fn is_var_pred(tokens: &Vec<Token>) -> bool {
    !is_negated_pred(tokens) && tokens[0].flag == TokenFlag::Variable
}

pub fn normalize_match_phrase(variable_token: &Token, mut match_phrase: Vec<Token>) -> Vec<Token> {
    let len = match_phrase.len();

    if len == 1 {
        match_phrase[0].is_negated = variable_token.is_negated;
        match_phrase[0].open_depth = variable_token.open_depth;
        match_phrase[len - 1].close_depth = variable_token.close_depth;
    } else {
        match_phrase[0].is_negated = variable_token.is_negated;
        if variable_token.open_depth > 0 {
            match_phrase[0].open_depth += variable_token.open_depth
        }
        if variable_token.close_depth > 0 {
            match_phrase[len - 1].close_depth += variable_token.close_depth
        }
    }

    match_phrase
}

pub fn build_phrase(phrase: &Phrase, string_cache: &StringCache) -> String {
    let mut tokens = vec![];

    for t in phrase {
        let mut string = String::new();

        if let Some(s) = t.as_str(string_cache) {
            if s.chars().any(|c| c.is_whitespace()) {
                string += &format!("`{}`", s);
            } else {
                string += s;
            }
        } else {
            string += &t.as_number().expect("number").to_string();
        }

        tokens.push(format!(
            "{}{}{}{}{}",
            String::from("(").repeat(t.open_depth as usize),
            if !t.is_consuming { "?" } else { "" },
            if t.is_negated {
                "!"
            } else if t.flag == TokenFlag::Side {
                "^"
            } else {
                ""
            },
            string,
            String::from(")").repeat(t.close_depth as usize)
        ));
    }

    tokens.join(" ")
}
