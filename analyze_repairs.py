import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys

def analyze_repairs(file_name):
    """
    Loads the cleaned repair data and generates three key visualizations.
    """
    print(f"Loading cleaned data from: {file_name}")
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        print("Please make sure this script is in the same folder as the CSV file.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    print("Data loaded successfully. Generating visualizations...")

    # --- Visualization 1: Top Repair Types ---

    # Get value counts for 'global_labor_code_description'
    repair_counts = df['global_labor_code_description'].value_counts()

    # We only want to plot the top 10 for readability
    top_n = 10
    repair_counts_top = repair_counts.head(top_n)

    # Plot a horizontal bar chart
    plt.figure(figsize=(10, 6))
    repair_counts_top.sort_values(ascending=True).plot(kind='barh', color='skyblue')

    plt.title(f'Top {top_n} Most Common Repair Types', fontsize=16)
    plt.xlabel('Number of Repairs', fontsize=12)
    plt.ylabel('Repair Type', fontsize=12)

    # Ensure x-axis shows whole numbers
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    output_png_1 = 'top_repair_types.png'
    plt.savefig(output_png_1)
    print(f"Saved: '{output_png_1}'")

    # --- Visualization 2: Repairs by Vehicle Platform ---

    # Get value counts for 'platform'
    platform_counts = df['platform'].value_counts()

    # Plot a horizontal bar chart
    plt.figure(figsize=(10, 6))
    platform_counts.sort_values(ascending=True).plot(kind='barh', color='coral')

    plt.title('Repairs by Vehicle Platform', fontsize=16)
    plt.xlabel('Number of Repairs', fontsize=12)
    plt.ylabel('Platform', fontsize=12)

    # Ensure x-axis shows whole numbers
    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    output_png_2 = 'repairs_by_platform.png'
    plt.savefig(output_png_2)
    print(f"Saved: '{output_png_2}'")

    # --- Visualization 3: Distribution of Total Repair Costs ---

    # Plot a histogram for 'totalcost'
    plt.figure(figsize=(10, 6))
    # Check if 'totalcost' column exists
    if 'totalcost' in df.columns:
        df['totalcost'].plot(kind='hist', bins=20, color='green', edgecolor='black')

        plt.title('Distribution of Total Repair Costs', fontsize=16)
        plt.xlabel('Total Cost ($)', fontsize=12)
        plt.ylabel('Frequency (Number of Repairs)', fontsize=12)

        # Format x-axis as currency
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('$%1.0f'))

        plt.tight_layout()
        output_png_3 = 'cost_distribution.png'
        plt.savefig(output_png_3)
        print(f"Saved: '{output_png_3}'")
    else:
        print("Warning: 'totalcost' column not found. Skipping cost distribution chart.")

    print("\nAnalysis complete. All charts have been saved as .png files.")

# --- Main execution ---
if __name__ == "__main__":
    # Use the file from the last successful step
    cleaned_file = 'cleaned_vehicle_repairs_thresh_5.csv'
    analyze_repairs(cleaned_file)