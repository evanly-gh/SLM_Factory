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
import static io.noties.prism4j.Prism4j.grammar;
import static io.noties.prism4j.Prism4j.pattern;
import static io.noties.prism4j.Prism4j.token;

import org.jetbrains.annotations.NotNull;

import io.noties.prism4j.Prism4j;
import io.noties.prism4j.annotations.Aliases;

@SuppressWarnings("unused")
@Aliases("jsonp")
public class Prism_json {

    @NotNull
    public static Prism4j.Grammar create(@NotNull Prism4j prism4j) {
        return grammar(
                "json",
                token("property", pattern(compile("\"(?:\\\\.|[^\\\\\"\\r\\n])*\"(?=\\s*:)", CASE_INSENSITIVE))),
                token("string", pattern(compile("\"(?:\\\\.|[^\\\\\"\\r\\n])*\"(?!\\s*:)"), false, true)),
                token("number", pattern(compile("\\b0x[\\dA-Fa-f]+\\b|(?:\\b\\d+\\.?\\d*|\\B\\.\\d+)(?:[Ee][+-]?\\d+)?"))),
                token("punctuation", pattern(compile("[{}\\[\\]);,]"))),
                // not sure about this one...
                token("operator", pattern(compile(":"))),
                token("boolean", pattern(compile("\\b(?:true|false)\\b", CASE_INSENSITIVE))),
                token("null", pattern(compile("\\bnull\\b", CASE_INSENSITIVE)))
        );
    }
}
