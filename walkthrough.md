# SmartPolicy Application Update: Currency Conversion

## Changes Implemented

### Currency Display
- **Symbol Change**: Refactored the application to display the Indian Rupee symbol (**₹**) instead of the US Dollar symbol (**$**).
- **Conversion Logic**: Implemented a global conversion rate of `1 USD = 86.0 INR`.
- **Locations Updated**:
    - **Main Prediction Result**: The predicted insurance charge now shows in Rupees.
    - **Monthly Estimate**: The monthly breakdown is also converted to Rupees.
    - **PDF Report**: The downloadable PDF report now includes the converted values and the correct currency symbol.

## Verification
- **Syntax Check**: Verified `app.py` for syntax errors.
- **Code Review**: Confirmed that the global `USD_TO_INR` constant is correctly applied in both the Streamlit UI and the PDF generation logic.

## Next Steps
- The application is running. You can interact with it to see the updated currency values in the "🔮 Prediction" tab.
