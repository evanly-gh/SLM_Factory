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
import static java.util.regex.Pattern.MULTILINE;
import static java.util.regex.Pattern.compile;
import static io.noties.prism4j.Prism4j.grammar;
import static io.noties.prism4j.Prism4j.pattern;
import static io.noties.prism4j.Prism4j.token;

import org.jetbrains.annotations.NotNull;

import io.noties.prism4j.GrammarUtils;
import io.noties.prism4j.Prism4j;
import io.noties.prism4j.annotations.Extend;

@SuppressWarnings("unused")
@Extend("clike")
public class Prism_java {

    @NotNull
    public static Prism4j.Grammar create(@NotNull Prism4j prism4j) {

        final Prism4j.Token keyword = token("keyword", pattern(compile("\\b(?:abstract|continue|for|new|switch|assert|default|goto|package|synchronized|boolean|do|if|private|this|break|double|implements|protected|throw|byte|else|import|public|throws|case|enum|instanceof|return|transient|catch|extends|int|short|try|char|final|interface|static|void|class|finally|long|strictfp|volatile|const|float|native|super|while)\\b")));

        final Prism4j.Grammar java = GrammarUtils.extend(GrammarUtils.require(prism4j, "clike"), "java",
                keyword,
                token("number", pattern(compile("\\b0b[01]+\\b|\\b0x[\\da-f]*\\.?[\\da-fp-]+\\b|(?:\\b\\d+\\.?\\d*|\\B\\.\\d+)(?:e[+-]?\\d+)?[df]?", CASE_INSENSITIVE))),
                token("operator", pattern(
                        compile("(^|[^.])(?:\\+[+=]?|-[-=]?|!=?|<<?=?|>>?>?=?|==?|&[&=]?|\\|[|=]?|\\*=?|\\/=?|%=?|\\^=?|[?:~])", MULTILINE),
                        true
                ))
        );

        GrammarUtils.insertBeforeToken(java, "function",
                token("annotation", pattern(
                        compile("(^|[^.])@\\w+"),
                        true,
                        false,
                        "punctuation"
                ))
        );

        GrammarUtils.insertBeforeToken(java, "class-name",
                token("generics", pattern(
                        compile("<\\s*\\w+(?:\\.\\w+)?(?:\\s*,\\s*\\w+(?:\\.\\w+)?)*>", CASE_INSENSITIVE),
                        false,
                        false,
                        "function",
                        grammar(
                                "inside",
                                keyword,
                                token("punctuation", pattern(compile("[<>(),.:]")))
                        )
                ))
        );

        return java;
    }
}
