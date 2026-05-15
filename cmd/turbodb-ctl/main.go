// Command turbodb-ctl is the control/operations CLI for TurboDB. It currently
// implements the Phase 4 import/export/inspect/convert tooling for the
// SafeTensors and GGUF tensor formats.
package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	cmd := os.Args[1]
	args := os.Args[2:]

	var err error
	switch cmd {
	case "import":
		err = runImport(args)
	case "export":
		err = runExport(args)
	case "inspect":
		err = runInspect(args)
	case "convert":
		err = runConvert(args)
	case "-h", "--help", "help":
		usage()
		return
	default:
		fmt.Fprintf(os.Stderr, "turbodb-ctl: unknown command %q\n\n", cmd)
		usage()
		os.Exit(2)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "turbodb-ctl %s: %v\n", cmd, err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `turbodb-ctl — TurboDB control CLI

Usage:
  turbodb-ctl import  --format safetensors|gguf --input FILE --collection NAME [--tensor NAME] [--data-dir DIR] [--bits N]
  turbodb-ctl export  --collection NAME --output FILE [--data-dir DIR]
  turbodb-ctl inspect --format safetensors|gguf --input FILE
  turbodb-ctl convert --from safetensors --to gguf --input FILE --output FILE

Examples:
  turbodb-ctl import --format safetensors --input embeddings.safetensors --collection docs
  turbodb-ctl import --format gguf --input model.gguf --tensor embedding.weight --collection vocab
  turbodb-ctl export --collection docs --output docs.safetensors
`)
}
