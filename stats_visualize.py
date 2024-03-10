import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from db import Base, Equity, Equity_Stats, Commodity, Commodity_Stats, MacroeconomicData, engine  # replace your_script_name

Session = sessionmaker(bind=engine)

def fetch_data(table, column_name, filter_column=None, filter_value=None):
    session = Session()
    try:
        query = session.query(getattr(table, column_name))
        if filter_column and filter_value:
            query = query.filter(getattr(table, filter_column) == filter_value)
        data = query.all()
        return [value[0] for value in data if value[0] is not None]
    finally:
        session.close()

def plot_histogram():
    table_name = table_var.get()
    column_name = column_var.get()
    num_bins = int(bin_entry.get())
    range_min = float(range_min_entry.get()) if range_min_entry.get() else None
    range_max = float(range_max_entry.get()) if range_max_entry.get() else None
    plot_range = (range_min, range_max) if range_min is not None and range_max is not None else None

    if table_name and column_name and num_bins > 0:
        table_class = table_classes[table_name]
        filter_column = None
        filter_value = None

        # Determine if a filter should be applied
        if table_name == 'Equity_Stats' and equity_stats_filter_var.get():
            filter_column = 'gics_sector'
            filter_value = equity_stats_filter_var.get()
        elif table_name == 'Commodity' and commodity_filter_var.get():
            filter_column = 'metal'
            filter_value = commodity_filter_var.get()
        elif table_name == 'Equity' and filter_var.get():
            filter_column = 'gics_sector'
            filter_value = filter_var.get()
        elif table_name == 'Commodity_Stats' and filter_var.get():
            filter_column = 'metal'
            filter_value = filter_var.get()
        elif table_name == 'MacroeconomicData' and filter_var.get():
            filter_column = 'gics_sector'
            filter_value = filter_var.get()
        else:
            filter_column = None
            filter_value = None

        # Fetch the data with the potential filter
        data = fetch_data(table_class, column_name, filter_column, filter_value)

        # Plotting the histogram
        fig, ax = plt.subplots()
        ax.hist(data, bins=num_bins, range=plot_range)
        ax.set_title(f'Histogram of {table_name} - {column_name}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

        # Display the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=window)  
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=8, column=0, columnspan=3)
        canvas.draw()
    else:
        print("Please enter valid inputs.")

def update_columns_and_filters(event):
    selected_table = table_var.get()
    # Update columns
    columns = table_columns.get(selected_table, [])
    column_menu['values'] = columns
    column_var.set('')

    # Update filter options
    if selected_table == 'Commodity_Stats':
        filter_var.set('')
        filter_menu['values'] = metals
        filter_label.config(text='Metal')
    elif selected_table == 'Commodity':
        filter_var.set('')
        filter_menu['values'] = metals
        filter_label.config(text='Metal:')
    else:
        filter_var.set('')
        filter_menu['values'] = []  # Reset to default
        filter_label.config(text='Filter:')

# GUI Setup
window = tk.Tk()
window.title("Database Histogram Plotter")

table_classes = {
    'Equity': Equity,
    'Equity_Stats': Equity_Stats,
    'Commodity': Commodity,
    'Commodity_Stats': Commodity_Stats,
    'MacroeconomicData': MacroeconomicData
}

table_columns = {
    'Equity': ['open', 'high', 'low', 'close', 'volume'],
    'Equity_Stats': ['change_day', 'atr'],
    'Commodity': ['open', 'high', 'low', 'close', 'volume'],
    'Commodity_Stats': ['change_day', 'atr'],
    'MacroeconomicData': ['value']
}

table_var = tk.StringVar()
column_var = tk.StringVar()

equity_stats_filter_var = tk.StringVar()
commodity_filter_var = tk.StringVar()

# Use a generic StringVar for the filter dropdown
filter_var = tk.StringVar()

# Define filter options
gics_sectors = ['Health Care', 'Financials', 'Technology']  # Add your GICS sectors here
metals = ['Gold', 'Copper', 'Crude Oil', 'Palladium', 'Platinum']  # Add your metals here

filter_label = tk.Label(window, text="Filter:")
filter_label.grid(row=5, column=0)
filter_var = tk.StringVar()
filter_menu = ttk.Combobox(window, textvariable=filter_var)
filter_menu.grid(row=5, column=1)

# Table selection
tk.Label(window, text="Select Table:").grid(row=0, column=0)
table_menu = ttk.Combobox(window, textvariable=table_var, values=list(table_classes.keys()))
table_menu.grid(row=0, column=1)

# Bind the update_columns_and_filters function to the table_menu selection event
table_menu.bind('<<ComboboxSelected>>', update_columns_and_filters)

# Column selection
tk.Label(window, text="Select Column:").grid(row=1, column=0)
column_menu = ttk.Combobox(window, textvariable=column_var)
column_menu.grid(row=1, column=1)

tk.Label(window, text="Number of Bins:").grid(row=2, column=0)
bin_entry = tk.Entry(window)
bin_entry.grid(row=2, column=1)

tk.Label(window, text="Range Min:").grid(row=3, column=0)
range_min_entry = tk.Entry(window)
range_min_entry.grid(row=3, column=1)

tk.Label(window, text="Range Max:").grid(row=4, column=0)
range_max_entry = tk.Entry(window)
range_max_entry.grid(row=4, column=1)

plot_button = tk.Button(window, text="Plot Histogram", command=plot_histogram)
plot_button.grid(row=5, column=0, columnspan=2)

window.mainloop()
