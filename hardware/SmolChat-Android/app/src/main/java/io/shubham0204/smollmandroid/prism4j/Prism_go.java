/*
 * Copyright (C) 2024 Shubham Panchal
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.shubham0204.smollmandroid.prism4j;

import static java.util.regex.Pattern.CASE_INSENSITIVE;
import static java.util.regex.Pattern.compile;
import static io.noties.prism4j.Prism4j.pattern;
import static io.noties.prism4j.Prism4j.token;

import org.jetbrains.annotations.NotNull;

import io.noties.prism4j.GrammarUtils;
import io.noties.prism4j.Prism4j;
import io.noties.prism4j.annotations.Extend;

@SuppressWarnings("unused")
@Extend("clike")
public class Prism_go {

    @NotNull
    public static Prism4j.Grammar create(@NotNull Prism4j prism4j) {

        final Prism4j.Grammar go = GrammarUtils.extend(
                GrammarUtils.require(prism4j, "clike"),
                "go",
                new GrammarUtils.TokenFilter() {
                    @Override
                    public boolean test(@NotNull Prism4j.Token token) {
                        return !"class-name".equals(token.name());
                    }
                },
                token("keyword", pattern(compile("\\b(?:break|case|chan|const|continue|default|defer|else|fallthrough|for|func|go(?:to)?|if|import|interface|map|package|range|return|select|struct|switch|type|var)\\b"))),
                token("boolean", pattern(compile("\\b(?:_|iota|nil|true|false)\\b"))),
                token("operator", pattern(compile("[*\\/%^!=]=?|\\+[=+]?|-[=-]?|\\|[=|]?|&(?:=|&|\\^=?)?|>(?:>=?|=)?|<(?:<=?|=|-)?|:=|\\.\\.\\."))),
                token("number", pattern(compile("(?:\\b0x[a-f\\d]+|(?:\\b\\d+\\.?\\d*|\\B\\.\\d+)(?:e[-+]?\\d+)?)i?", CASE_INSENSITIVE))),
                token("string", pattern(
                        compile("([\"'`])(\\\\[\\s\\S]|(?!\\1)[^\\\\])*\\1"),
                        false,
                        true
                ))
        );

        // clike doesn't have builtin
        GrammarUtils.insertBeforeToken(go, "boolean",
                token("builtin", pattern(compile("\\b(?:bool|byte|complex(?:64|128)|error|float(?:32|64)|rune|string|u?int(?:8|16|32|64)?|uintptr|append|cap|close|complex|copy|delete|imag|len|make|new|panic|print(?:ln)?|real|recover)\\b")))
        );

        return go;
    }
}
