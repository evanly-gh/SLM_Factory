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

import static java.util.regex.Pattern.compile;
import static io.noties.prism4j.Prism4j.grammar;
import static io.noties.prism4j.Prism4j.pattern;
import static io.noties.prism4j.Prism4j.token;

import org.jetbrains.annotations.NotNull;

import io.noties.prism4j.Prism4j;

@SuppressWarnings("unused")
public class Prism_brainfuck {

    @NotNull
    public static Prism4j.Grammar create(@NotNull Prism4j prism4j) {
        return grammar("brainfuck",
                token("pointer", pattern(compile("<|>"), false, false, "keyword")),
                token("increment", pattern(compile("\\+"), false, false, "inserted")),
                token("decrement", pattern(compile("-"), false, false, "deleted")),
                token("branching", pattern(compile("\\[|\\]"), false, false, "important")),
                token("operator", pattern(compile("[.,]"))),
                token("comment", pattern(compile("\\S+")))
        );
    }
}
