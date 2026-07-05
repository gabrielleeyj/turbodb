// TurboDB Control is the operator CLI for TurboDB (SCOPE Component 8).
//
// Remote commands (collection, index, sync reconcile) talk to a running
// turbodb-engine over gRPC via --engine. File commands (import, export,
// inspect, convert) open an engine data directory directly via --data-dir
// and must not run against a directory a live engine is using.
package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

// version is stamped via -ldflags "-X main.version=..." at release build.
var version = "dev"

func main() {
	if err := newRootCmd().Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "turbodb-ctl: %v\n", err)
		os.Exit(1)
	}
}

func newRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:           "turbodb-ctl",
		Short:         "Operator CLI for TurboDB",
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	root.PersistentFlags().String("engine", "localhost:7080", "engine gRPC address for remote commands")

	root.AddCommand(
		newAdminCmd(),
		newCollectionCmd(),
		newIndexCmd(),
		newImportCmd(),
		newExportCmd(),
		newInspectCmd(),
		newConvertCmd(),
		newSyncCmd(),
		newVersionCmd(),
	)
	return root
}

func newVersionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Print the turbodb-ctl version",
		Run: func(cmd *cobra.Command, _ []string) {
			fmt.Fprintln(cmd.OutOrStdout(), "turbodb-ctl", version)
		},
	}
}
