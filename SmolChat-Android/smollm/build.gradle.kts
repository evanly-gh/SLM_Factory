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

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "io.shubham0204.smollm"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    defaultConfig {
        minSdk = 26
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
        externalNativeBuild {
            cmake {
                cppFlags += listOf()
                // allow compiling 16 KB page-aligned shared libraries
                // https://developer.android.com/guide/practices/page-sizes#compile-r27
                arguments += listOf("-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON")
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                arguments += "-DCMAKE_MESSAGE_LOG_LEVEL=DEBUG"
                arguments += "-DCMAKE_VERBOSE_MAKEFILE=ON"

                arguments += "-DBUILD_SHARED_LIBS=ON"
                arguments += "-DLLAMA_BUILD_COMMON=ON"
                arguments += "-DLLAMA_CURL=OFF"
                arguments += "-DGGML_LLAMAFILE=OFF"
                // (debugging) uncomment the following line to enable debug builds
                // and attach hardware-assisted address sanitizer
                // arguments += "-DCMAKE_BUILD_TYPE=Debug"
                // arguments += listOf("-DANDROID_SANITIZE=hwaddress")
            }
        }
    }

    // "backend" flavors select the native build variant (see
    // src/main/cpp/CMakeLists.txt's SMOLLM_BUILD_VULKAN_VARIANT). "cpu" adds
    // no extra CMake arguments -- its native build invocation is identical
    // to what this module built before these flavors existed. "vulkan" is
    // additive only: a separate build directory, separate output, and (see
    // minSdk below) its own higher floor, none of which affects "cpu".
    // app/build.gradle.kts mirrors this same dimension/flavor pair so each
    // flavor's APK links against the matching :smollm variant.
    flavorDimensions += "backend"
    productFlavors {
        create("cpu") {
            dimension = "backend"
        }
        create("vulkan") {
            dimension = "backend"
            // vkGetPhysicalDeviceFeatures2 (Vulkan 1.1 core) is only present
            // in the NDK's stub libvulkan.so from API 28 onward -- confirmed
            // via a scratch build (see conversation), not a guess.
            minSdk = 28
            // GGML_VULKAN is a single global CMake option for this whole
            // configure pass (see CMakeLists.txt) -- it's not scoped per-ABI,
            // so without this filter Gradle would still invoke CMake for
            // every other ABI too, and ggml-vulkan.cpp genuinely does not
            // compile for 32-bit targets (Vulkan-Hpp deliberately keeps
            // handle-to-raw-pointer conversions explicit there -- confirmed
            // via a real build attempt, not a guess). build_library_vulkan()
            // is arm64-v8a-only anyway (see CMakeLists.txt), so this filter
            // just stops the other ABIs' CMake invocations from happening at
            // all, rather than relying on the CMakeLists.txt guard alone.
            ndk {
                abiFilters += "arm64-v8a"
            }
            externalNativeBuild {
                cmake {
                    arguments += "-DSMOLLM_BUILD_VULKAN_VARIANT=ON"
                }
            }
        }
        create("opencl") {
            dimension = "backend"
            // Unlike Vulkan, there's no NDK-stub API-level gate here (Android
            // ships no NDK OpenCL headers/stub at all -- vendor/opencl-headers
            // and vendor/opencl-icd-loader supply our own), so no minSdk floor
            // beyond the module's existing default is needed.
            // GGML_OPENCL is a single global CMake option for this whole
            // configure pass (see CMakeLists.txt), same as GGML_VULKAN -- and
            // build_library_opencl() is arm64-v8a-only, so this filter stops
            // Gradle from invoking CMake for other ABIs at all, matching the
            // "vulkan" flavor's own reasoning above.
            ndk {
                abiFilters += "arm64-v8a"
            }
            externalNativeBuild {
                cmake {
                    arguments += "-DSMOLLM_BUILD_OPENCL_VARIANT=ON"
                }
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.1")
    testImplementation(libs.junit)
    androidTestImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation(libs.androidx.junit)
}
