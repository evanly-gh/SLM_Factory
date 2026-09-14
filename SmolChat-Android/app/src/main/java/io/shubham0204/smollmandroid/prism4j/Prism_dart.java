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
import static io.noties.prism4j.Prism4j.pattern;
import static io.noties.prism4j.Prism4j.token;

import org.jetbrains.annotations.NotNull;

import io.noties.prism4j.GrammarUtils;
import io.noties.prism4j.Prism4j;
import io.noties.prism4j.annotations.Extend;

@SuppressWarnings("unused")
@Extend("clike")
public class Prism_dart {

    @NotNull
    public static Prism4j.Grammar create(@NotNull Prism4j prism4j) {

        final Prism4j.Grammar dart = GrammarUtils.extend(
                GrammarUtils.require(prism4j, "clike"),
                "dart",
                token("string",
                        pattern(compile("r?(\"\"\"|''')[\\s\\S]*?\\1"), false, true),
                        pattern(compile("r?(\"|')(?:\\\\.|(?!\\1)[^\\\\\\r\\n])*\\1"), false, true)
                ),
                token("keyword",
                        pattern(compile("\\b(?:async|sync|yield)\\*")),
                        pattern(compile("\\b(?:abstract|assert|async|await|break|case|catch|class|const|continue|default|deferred|do|dynamic|else|enum|export|external|extends|factory|final|finally|for|get|if|implements|import|in|library|new|null|operator|part|rethrow|return|set|static|super|switch|this|throw|try|typedef|var|void|while|with|yield)\\b"))
                ),
                token("operator", pattern(compile("\\bis!|\\b(?:as|is)\\b|\\+\\+|--|&&|\\|\\||<<=?|>>=?|~(?:\\/=?)?|[+\\-*\\/%&^|=!<>]=?|\\?")))
        );

        GrammarUtils.insertBeforeToken(dart, "function",
                token("metadata", pattern(compile("@\\w+"), false, false, "symbol"))
        );

        return dart;
    }
}
