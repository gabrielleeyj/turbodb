package codebook

import "embed"

// precomputedFS holds embedded precomputed codebook JSON files.
// The embed pattern uses "all:" prefix to allow an empty directory.
//
//go:embed all:precomputed
var precomputedFS embed.FS
