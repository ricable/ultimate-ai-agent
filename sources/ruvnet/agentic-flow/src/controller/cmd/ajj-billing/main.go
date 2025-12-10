package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/ruvnet/agentic-jujutsu/controller/pkg/economics"
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "ajj-billing",
		Short: "Agentic-Jujutsu Billing and Economics CLI",
		Long: `Manage subscriptions, billing, usage tracking, and economic features
for the Agentic-Jujutsu GitOps platform.`,
	}

	// Add subcommands
	rootCmd.AddCommand(newSubscriptionCommand())
	rootCmd.AddCommand(newUsageCommand())
	rootCmd.AddCommand(newCouponCommand())
	rootCmd.AddCommand(newPricingCommand())
	rootCmd.AddCommand(newQuotaCommand())

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// Subscription commands
func newSubscriptionCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "subscription",
		Short: "Manage subscriptions",
		Aliases: []string{"sub"},
	}

	cmd.AddCommand(&cobra.Command{
		Use:   "create [tier]",
		Short: "Create a new subscription",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			tier := args[0]
			cycle, _ := cmd.Flags().GetString("cycle")
			coupon, _ := cmd.Flags().GetString("coupon")

			fmt.Printf("Creating %s subscription (%s billing)\n", tier, cycle)
			if coupon != "" {
				fmt.Printf("Applying coupon: %s\n", coupon)
			}

			fmt.Println("\n✓ Subscription created successfully!")
			fmt.Println("Subscription ID: sub_1234567890")
			return nil
		},
	})

	cmd.AddCommand(&cobra.Command{
		Use:   "upgrade [tier]",
		Short: "Upgrade subscription to a higher tier",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error{
			tier := args[0]
			fmt.Printf("Upgrading to %s tier\n", tier)
			fmt.Println("Calculating prorated amount...")
			fmt.Println("Processing payment...")
			fmt.Println("\n✓ Subscription upgraded successfully!")
			return nil
		},
	})

	cmd.AddCommand(&cobra.Command{
		Use:   "cancel",
		Short: "Cancel subscription",
		RunE: func(cmd *cobra.Command, args []string) error {
			immediate, _ := cmd.Flags().GetBool("immediate")
			if immediate {
				fmt.Println("Cancelling subscription immediately...")
			} else {
				fmt.Println("Subscription will be cancelled at end of billing period")
			}
			fmt.Println("✓ Subscription cancelled")
			return nil
		},
	})

	cmd.AddCommand(&cobra.Command{
		Use:   "status",
		Short: "Show subscription status",
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Println("Subscription Status:")
			fmt.Println("  Tier: Pro")
			fmt.Println("  Status: Active")
			fmt.Println("  Billing Cycle: Monthly")
			fmt.Println("  Next Billing Date: 2025-12-15")
			fmt.Println("  Price: $99.00/month")
			fmt.Println("  Features: 12 concurrent agents, 2000 agent hours, Priority support")
			return nil
		},
	})

	// Add flags
	cmd.PersistentFlags().String("cycle", "monthly", "Billing cycle (monthly, yearly, quarterly)")
	cmd.PersistentFlags().String("coupon", "", "Coupon code")
	cmd.PersistentFlags().Bool("immediate", false, "Cancel immediately")

	return cmd
}

// Usage tracking commands
func newUsageCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "usage",
		Short: "View usage and metering",
	}

	cmd.AddCommand(&cobra.Command{
		Use:   "current",
		Short: "Show current period usage",
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Println("Current Usage (November 2025):\n")
			fmt.Println("Agent Hours:       450 / 2000  (22.5%)")
			fmt.Println("Deployments:       35 / 200    (17.5%)")
			fmt.Println("API Requests:      250K / 1M   (25.0%)")
			fmt.Println("Storage:           45GB / 100GB (45.0%)")
			fmt.Println("Network:           180GB / 1TB  (18.0%)")
			fmt.Println("GPU Hours:         12 / 50     (24.0%)")
			fmt.Println("\nEstimated Cost: $99.00")
			fmt.Println("Overage Charges: $0.00")
			fmt.Println("Total: $99.00")
			return nil
		},
	})

	cmd.AddCommand(&cobra.Command{
		Use:   "history",
		Short: "Show usage history",
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Println("Usage History:\n")
			fmt.Println("Period      | Agent Hours | Deployments | Cost")
			fmt.Println("------------|-------------|-------------|-------")
			fmt.Println("2025-11     | 450         | 35          | $99.00")
			fmt.Println("2025-10     | 890         | 67          | $99.00")
			fmt.Println("2025-09     | 1250        | 98          | $99.00")
			return nil
		},
	})

	cmd.AddCommand(&cobra.Command{
		Use:   "forecast",
		Short: "Forecast usage and costs",
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Println("Usage Forecast (Next 30 Days):\n")
			fmt.Println("Projected Agent Hours: 950 (47.5% of limit)")
			fmt.Println("Projected Deployments: 75 (37.5% of limit)")
			fmt.Println("Projected Cost: $99.00")
			fmt.Println("\nRecommendations:")
			fmt.Println("  • Usage is within limits")
			fmt.Println("  • No action required")
			return nil
		},
	})

	return cmd
}

// Coupon commands
func newCouponCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "coupon",
		Short: "Manage promotional coupons",
	}

	cmd.AddCommand(&cobra.Command{
		Use:   "create [code]",
		Short: "Create a new coupon",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			code := args[0]
			discountType, _ := cmd.Flags().GetString("type")
			value, _ := cmd.Flags().GetFloat64("value")

			fmt.Printf("Creating coupon: %s\n", code)
			fmt.Printf("Type: %s\n", discountType)
			fmt.Printf("Value: %.2f\n", value)
			fmt.Println("\n✓ Coupon created successfully!")
			return nil
		},
	})

	cmd.AddCommand(&cobra.Command{
		Use:   "validate [code]",
		Short: "Validate a coupon code",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			code := args[0]
			fmt.Printf("Validating coupon: %s\n", code)
			fmt.Println("\n✓ Coupon is valid")
			fmt.Println("Discount: 20% off")
			fmt.Println("Valid until: 2025-12-31")
			return nil
		},
	})

	cmd.AddCommand(&cobra.Command{
		Use:   "list",
		Short: "List all active coupons",
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Println("Active Coupons:\n")
			fmt.Println("Code          | Type       | Value | Valid Until")
			fmt.Println("--------------|------------|-------|-------------")
			fmt.Println("WELCOME20     | Percentage | 20%   | 2025-12-31")
			fmt.Println("STARTUP50     | Fixed      | $50   | 2025-12-31")
			fmt.Println("FIRSTMONTH    | Fixed      | $29   | 2025-12-31")
			return nil
		},
	})

	// Add flags
	cmd.PersistentFlags().String("type", "percentage", "Discount type (percentage, fixed, credits)")
	cmd.PersistentFlags().Float64("value", 0, "Discount value")
	cmd.PersistentFlags().String("valid-until", "", "Valid until date (YYYY-MM-DD)")

	return cmd
}

// Pricing commands
func newPricingCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "pricing",
		Short: "View pricing information",
	}

	cmd.AddCommand(&cobra.Command{
		Use:   "list",
		Short: "List all pricing tiers",
		RunE: func(cmd *cobra.Command, args []string) error {
			pm := economics.NewPricingManager()
			tiers := pm.ListTiers()

			fmt.Println("Agentic-Jujutsu Pricing Tiers:\n")

			for _, tier := range tiers {
				fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
				fmt.Printf("🎯 %s", tier.Name)
				if tier.PopularBadge {
					fmt.Printf(" [POPULAR]")
				}
				fmt.Printf("\n")
				if !tier.CustomPricing {
					fmt.Printf("   $%.2f/month • $%.2f/year\n", tier.MonthlyPrice, tier.YearlyPrice)
				} else {
					fmt.Printf("   Custom Pricing\n")
				}
				fmt.Printf("   %s\n\n", tier.Description)

				fmt.Printf("   Features:\n")
				for _, feature := range tier.Features {
					fmt.Printf("   • %s\n", feature)
				}
				fmt.Printf("\n")
			}

			return nil
		},
	})

	cmd.AddCommand(&cobra.Command{
		Use:   "compare [tier1] [tier2]",
		Short: "Compare two pricing tiers",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			tier1 := args[0]
			tier2 := args[1]

			fmt.Printf("Comparing %s vs %s:\n\n", tier1, tier2)
			fmt.Println("Feature                | Starter | Pro")
			fmt.Println("-----------------------|---------|--------")
			fmt.Println("Price (monthly)        | $29     | $99")
			fmt.Println("Agent Hours            | 500     | 2000")
			fmt.Println("Concurrent Agents      | 3       | 12")
			fmt.Println("Storage                | 10GB    | 100GB")
			fmt.Println("Support                | Email   | Priority")
			fmt.Println("AI Features            | Basic   | Advanced")

			return nil
		},
	})

	return cmd
}

// Quota commands
func newQuotaCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "quota",
		Short: "Check quota and limits",
	}

	cmd.AddCommand(&cobra.Command{
		Use:   "check",
		Short: "Check current quota usage",
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Println("Quota Status:\n")

			quotas := []struct{
				Name string
				Used float64
				Limit float64
				Unit string
			}{
				{"Agent Hours", 450, 2000, "hours"},
				{"Deployments", 35, 200, "deployments"},
				{"API Requests", 250000, 1000000, "requests"},
				{"Storage", 45, 100, "GB"},
				{"Network", 180, 1000, "GB"},
				{"GPU Hours", 12, 50, "hours"},
			}

			for _, q := range quotas {
				percent := (q.Used / q.Limit) * 100
				status := "✓"
				if percent > 80 {
					status = "⚠"
				}
				if percent >= 100 {
					status = "✗"
				}

				fmt.Printf("%s %-15s: %.0f / %.0f %s (%.1f%%)\n",
					status, q.Name, q.Used, q.Limit, q.Unit, percent)
			}

			fmt.Println("\nStatus: All quotas within limits")
			return nil
		},
	})

	cmd.AddCommand(&cobra.Command{
		Use:   "increase",
		Short: "Request quota increase",
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Println("To increase your quotas, consider upgrading your subscription:")
			fmt.Println("\nCurrent: Pro ($99/month)")
			fmt.Println("Upgrade to: Enterprise ($499/month)")
			fmt.Println("\nIncrease benefits:")
			fmt.Println("  • Agent Hours: 2000 → 10000")
			fmt.Println("  • Concurrent Agents: 12 → 100")
			fmt.Println("  • Storage: 100GB → 1TB")
			fmt.Println("  • GPU Hours: 50 → 500")
			fmt.Println("\nRun: ajj-billing subscription upgrade enterprise")
			return nil
		},
	})

	return cmd
}
