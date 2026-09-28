import pandas as pd
import numpy as np
import os


#=========================================================
#STEP 1: DEFINE FILE LOCATIONS
#=========================================================

#Folder containing Python model outputs
results_folder = "results"


#Final Excel workbook location
output_file = os.path.join(
    results_folder,
    "Automobile_Insurance_Pricing_Analysis.xlsx"
)


#=========================================================
#STEP 2: LOAD PYTHON MODEL OUTPUTS
#=========================================================

print("Loading model results...")


#Load policy-level pricing results
policy_results = pd.read_csv(
    os.path.join(
        results_folder,
        "policy_pricing_results.csv"
    )
)


#Load segment pricing results
driver_age = pd.read_csv(
    os.path.join(
        results_folder,
        "driver_age_pricing.csv"
    )
)

vehicle_age = pd.read_csv(
    os.path.join(
        results_folder,
        "vehicle_age_pricing.csv"
    )
)

bonus_malus = pd.read_csv(
    os.path.join(
        results_folder,
        "bonus_malus_pricing.csv"
    )
)

density = pd.read_csv(
    os.path.join(
        results_folder,
        "density_pricing.csv"
    )
)


#Load holdout validation results
validation_deciles = pd.read_csv(
    os.path.join(
        results_folder,
        "model_validation_deciles.csv"
    )
)

validation_policy = pd.read_csv(
    os.path.join(
        results_folder,
        "model_validation_policy_results.csv"
    )
)


#Load the validation summary exported by 05_model_validation.py
validation_summary_path = os.path.join(
    results_folder,
    "model_validation_summary.csv"
)


if not os.path.exists(validation_summary_path):

    raise FileNotFoundError(
        "\nMissing results/model_validation_summary.csv.\n"
        "Add the summary export block to the bottom of "
        "05_model_validation.py, run that file again, "
        "and then rerun 07_build_excel_dashboard.py."
    )


validation_summary = pd.read_csv(
    validation_summary_path
).iloc[0]


print("Model results loaded successfully")


#=========================================================
#STEP 3: CALCULATE FULL-PORTFOLIO MODEL VALUES
#=========================================================

#Calculate portfolio totals directly from policy-level predictions
total_policies = len(
    policy_results
)

total_exposure = (
    policy_results["Exposure"].sum()
)

predicted_total_claims = (
    policy_results["ExpectedClaims"].sum()
)

predicted_total_losses = (
    policy_results["ExpectedLossAmount"].sum()
)


#=========================================================
#STEP 4: CALCULATE VALIDATION VALUES
#=========================================================

#Calculate holdout values directly from policy-level validation data
validation_test_policies = len(
    validation_policy
)

validation_exposure = (
    validation_policy["Exposure"].sum()
)

validation_actual_claims = (
    validation_policy["ClaimNb"].sum()
)

validation_predicted_claims = (
    validation_policy["PredictedClaims"].sum()
)

validation_actual_losses = (
    validation_policy["ClaimAmount"].sum()
)

validation_predicted_losses = (
    validation_policy["PredictedLossAmount"].sum()
)


#Read metrics that require the model-validation run
baseline_tweedie_deviance = float(
    validation_summary["BaselineTweedieDeviance"]
)

model_tweedie_deviance = float(
    validation_summary["ModelTweedieDeviance"]
)

ordered_gini = float(
    validation_summary["OrderedGini"]
)


#=========================================================
#STEP 5: CREATE THE EXCEL WORKBOOK
#=========================================================

print("\nCreating Excel pricing workbook...")


with pd.ExcelWriter(
        output_file,
        engine="xlsxwriter"
) as writer:

    workbook = writer.book


    #=====================================================
    #WORKBOOK COLOR PALETTE
    #=====================================================

    navy = "#17365D"
    blue = "#2F75B5"
    orange = "#ED7D31"
    pale_blue = "#F4F7FA"
    pale_yellow = "#FFF2CC"
    pale_orange = "#FCE4D6"
    gray_text = "#666666"
    note_text = "#7F6000"

    chart_blue = "#156082"
    chart_orange = "#E97132"
    grid_gray = "#CCCCCC"
    chart_border = "#D9D9D9"

    green_cf = "#DCFCE7"
    yellow_cf = "#FEF3C7"
    red_cf = "#FECACA"


    #=====================================================
    #WORKBOOK FORMATS
    #=====================================================

    title_format = workbook.add_format(
        {
            "font_name": "Carlito",
            "bold": True,
            "font_size": 14,
            "font_color": "#FFFFFF",
            "bg_color": navy,
            "align": "center",
            "valign": "vcenter"
        }
    )


    dashboard_title_format = workbook.add_format(
        {
            "font_name": "Carlito",
            "bold": True,
            "font_size": 18,
            "font_color": "#FFFFFF",
            "bg_color": navy,
            "align": "center",
            "valign": "vcenter"
        }
    )


    table_header_format = workbook.add_format(
        {
            "font_name": "Carlito",
            "bold": True,
            "font_size": 11,
            "font_color": "#FFFFFF",
            "bg_color": blue,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True
        }
    )


    section_blue_format = workbook.add_format(
        {
            "font_name": "Carlito",
            "bold": True,
            "font_size": 12,
            "font_color": "#FFFFFF",
            "bg_color": blue,
            "align": "left",
            "valign": "vcenter"
        }
    )


    section_orange_format = workbook.add_format(
        {
            "font_name": "Carlito",
            "bold": True,
            "font_size": 12,
            "font_color": "#FFFFFF",
            "bg_color": orange,
            "align": "left",
            "valign": "vcenter"
        }
    )


    dashboard_label_format = workbook.add_format(
        {
            "font_name": "Carlito",
            "bold": True,
            "font_size": 11,
            "font_color": gray_text,
            "align": "center_across",
            "valign": "vcenter"
        }
    )


    portfolio_kpi_integer = workbook.add_format(
        {
            "font_name": "Carlito",
            "bold": True,
            "font_size": 16,
            "font_color": navy,
            "bg_color": pale_blue,
            "align": "center",
            "valign": "vcenter",
            "num_format": "#,##0"
        }
    )


    portfolio_kpi_decimal = workbook.add_format(
        {
            "font_name": "Carlito",
            "bold": True,
            "font_size": 16,
            "font_color": navy,
            "bg_color": pale_blue,
            "align": "center",
            "valign": "vcenter",
            "num_format": "#,##0.00"
        }
    )


    portfolio_kpi_percent = workbook.add_format(
        {
            "font_name": "Carlito",
            "bold": True,
            "font_size": 16,
            "font_color": navy,
            "bg_color": pale_blue,
            "align": "center",
            "valign": "vcenter",
            "num_format": "0.00%"
        }
    )


    portfolio_kpi_currency = workbook.add_format(
        {
            "font_name": "Carlito",
            "bold": True,
            "font_size": 16,
            "font_color": navy,
            "bg_color": pale_blue,
            "align": "center",
            "valign": "vcenter",
            "num_format": r"\€#,##0.00"
        }
    )


    validation_kpi_currency = workbook.add_format(
        {
            "font_name": "Carlito",
            "bold": True,
            "font_size": 16,
            "font_color": navy,
            "bg_color": pale_yellow,
            "align": "center",
            "valign": "vcenter",
            "num_format": r"\€#,##0.00"
        }
    )


    validation_kpi_percent = workbook.add_format(
        {
            "font_name": "Carlito",
            "bold": True,
            "font_size": 16,
            "font_color": navy,
            "bg_color": pale_yellow,
            "align": "center",
            "valign": "vcenter",
            "num_format": "0.00%"
        }
    )


    validation_kpi_decimal = workbook.add_format(
        {
            "font_name": "Carlito",
            "bold": True,
            "font_size": 16,
            "font_color": navy,
            "bg_color": pale_yellow,
            "align": "center",
            "valign": "vcenter",
            "num_format": "0.0000"
        }
    )


    note_format = workbook.add_format(
        {
            "font_name": "Carlito",
            "font_size": 11,
            "font_color": note_text,
            "bg_color": pale_orange,
            "text_wrap": True,
            "valign": "vcenter"
        }
    )


    body_text = workbook.add_format(
        {
            "font_name": "Carlito",
            "font_size": 11,
            "text_wrap": True,
            "valign": "top"
        }
    )


    integer_format = workbook.add_format(
        {
            "font_name": "Carlito",
            "font_size": 11,
            "num_format": "#,##0"
        }
    )


    decimal_2_format = workbook.add_format(
        {
            "font_name": "Carlito",
            "font_size": 11,
            "num_format": "#,##0.00"
        }
    )


    decimal_4_format = workbook.add_format(
        {
            "font_name": "Carlito",
            "font_size": 11,
            "num_format": "0.0000"
        }
    )


    currency_format = workbook.add_format(
        {
            "font_name": "Carlito",
            "font_size": 11,
            "num_format": r"\€#,##0.00"
        }
    )


    percent_format = workbook.add_format(
        {
            "font_name": "Carlito",
            "font_size": 11,
            "num_format": "0.00%"
        }
    )


    relativity_format = workbook.add_format(
        {
            "font_name": "Carlito",
            "font_size": 11,
            "num_format": "0.000"
        }
    )


    #=====================================================
    #STEP 6: MODEL SUMMARY SHEET
    #=====================================================

    model_sheet = workbook.add_worksheet(
        "Model Summary"
    )


    model_sheet.set_column(
        "A:A",
        34
    )

    model_sheet.set_column(
        "B:B",
        16
    )

    model_sheet.set_column(
        "C:C",
        24
    )

    model_sheet.set_column(
        "D:D",
        40
    )


    model_sheet.merge_range(
        "A1:D1",
        "Automobile Insurance Pricing Analysis — Model Summary",
        title_format
    )


    model_sheet.write_row(
        "A3",
        [
            "Metric",
            "Value",
            "Unit / Scope",
            "Notes"
        ],
        table_header_format
    )


    model_rows = [
        [
            "Total policies",
            total_policies,
            "Full portfolio",
            "freMTPL2 policy records analyzed"
        ],
        [
            "Total exposure",
            total_exposure,
            "Policy-years",
            "Exposure capped at 1 for modeling"
        ],
        [
            "Predicted total claims",
            predicted_total_claims,
            "Full portfolio",
            "Final Poisson model"
        ],
        [
            "Predicted claim frequency",
            None,
            "Claims per policy-year",
            "Final Poisson model"
        ],
        [
            "Predicted average severity",
            None,
            "EUR per claim",
            "Final Gamma model"
        ],
        [
            "Predicted pure premium",
            None,
            "EUR per policy-year",
            "Frequency × severity"
        ],
        [
            "Predicted total losses",
            predicted_total_losses,
            "EUR",
            "Full-portfolio model estimate"
        ],
        [
            "Validation test policies",
            validation_test_policies,
            "Holdout set",
            "20% matched-data test sample"
        ],
        [
            "Validation actual pure premium",
            None,
            "EUR per policy-year",
            "Matched frequency/severity holdout"
        ],
        [
            "Validation predicted pure premium",
            None,
            "EUR per policy-year",
            "Matched frequency/severity holdout"
        ],
        [
            "Validation loss prediction error",
            None,
            "Percent",
            "Predicted vs actual total loss"
        ],
        [
            "Baseline Tweedie deviance",
            baseline_tweedie_deviance,
            "Holdout validation",
            "Constant pure-premium baseline"
        ],
        [
            "Model Tweedie deviance",
            model_tweedie_deviance,
            "Holdout validation",
            "Combined frequency × severity model"
        ],
        [
            "Tweedie deviance improvement",
            None,
            "Percent",
            "Calculated from baseline and model deviance"
        ],
        [
            "Ordered Gini coefficient",
            ordered_gini,
            "Holdout validation",
            "Positive discrimination / risk ranking"
        ]
    ]


    for excel_row, row_data in enumerate(
            model_rows,
            start=4
    ):

        model_sheet.write(
            excel_row - 1,
            0,
            row_data[0],
            body_text
        )

        if row_data[1] is not None:

            model_sheet.write(
                excel_row - 1,
                1,
                row_data[1]
            )

        model_sheet.write(
            excel_row - 1,
            2,
            row_data[2],
            body_text
        )

        model_sheet.write(
            excel_row - 1,
            3,
            row_data[3],
            body_text
        )


    #Formula-driven portfolio metrics
    model_sheet.write_formula(
        "B7",
        "=B6/B5",
        decimal_4_format
    )

    model_sheet.write_formula(
        "B8",
        "=B10/B6",
        currency_format
    )

    model_sheet.write_formula(
        "B9",
        "=B10/B5",
        currency_format
    )


    #Formula-driven validation metrics
    model_sheet.write_formula(
        "B12",
        "='Validation'!E4+'Validation'!E5+'Validation'!E6+'Validation'!E7+'Validation'!E8+'Validation'!E9+'Validation'!E10+'Validation'!E11+'Validation'!E12+'Validation'!E13"
        "/('Validation'!C4+'Validation'!C5+'Validation'!C6+'Validation'!C7+'Validation'!C8+'Validation'!C9+'Validation'!C10+'Validation'!C11+'Validation'!C12+'Validation'!C13)",
        currency_format
    )

    model_sheet.write_formula(
        "B13",
        "='Validation'!G4+'Validation'!G5+'Validation'!G6+'Validation'!G7+'Validation'!G8+'Validation'!G9+'Validation'!G10+'Validation'!G11+'Validation'!G12+'Validation'!G13"
        "/('Validation'!C4+'Validation'!C5+'Validation'!C6+'Validation'!C7+'Validation'!C8+'Validation'!C9+'Validation'!C10+'Validation'!C11+'Validation'!C12+'Validation'!C13)",
        currency_format
    )

    model_sheet.write_formula(
        "B14",
        "=(SUM('Validation'!G4:G13)-SUM('Validation'!E4:E13))/SUM('Validation'!E4:E13)",
        percent_format
    )

    model_sheet.write(
        "B15",
        baseline_tweedie_deviance,
        decimal_4_format
    )

    model_sheet.write(
        "B16",
        model_tweedie_deviance,
        decimal_4_format
    )

    model_sheet.write_formula(
        "B17",
        "=(B15-B16)/B15",
        percent_format
    )

    model_sheet.write(
        "B18",
        ordered_gini,
        decimal_4_format
    )


    #Apply appropriate number formats to direct-input values
    model_sheet.write(
        "B4",
        total_policies,
        integer_format
    )

    model_sheet.write(
        "B5",
        total_exposure,
        decimal_2_format
    )

    model_sheet.write(
        "B6",
        predicted_total_claims,
        decimal_2_format
    )

    model_sheet.write(
        "B10",
        predicted_total_losses,
        currency_format
    )

    model_sheet.write(
        "B11",
        validation_test_policies,
        integer_format
    )


    model_sheet.freeze_panes(
        3,
        0
    )


    #=====================================================
    #STEP 7: SEGMENT PRICING SHEET
    #=====================================================

    segment_sheet = workbook.add_worksheet(
        "Segment Pricing"
    )


    segment_sheet.set_column(
        "A:A",
        18
    )

    segment_sheet.set_column(
        "B:B",
        12
    )

    segment_sheet.set_column(
        "C:D",
        14
    )

    segment_sheet.set_column(
        "E:F",
        16
    )

    segment_sheet.set_column(
        "G:H",
        18
    )

    segment_sheet.set_column(
        "I:I",
        16
    )


    segment_sheet.merge_range(
        "A1:I1",
        "Model-Based Segment Pricing Relativities",
        title_format
    )


    segment_headers = [
        "Driver Age Group",
        "Policies",
        "Exposure",
        "Expected Claims",
        "Expected Losses",
        "Predicted Frequency",
        "Predicted Severity",
        "Predicted Pure Premium",
        "Pricing Relativity"
    ]


    #-----------------------------------------------------
    #Helper Function for Segment Tables
    #-----------------------------------------------------

    def write_segment_table(
            dataframe,
            header_row,
            first_data_row,
            label_header
    ):

        headers = segment_headers.copy()
        headers[0] = label_header

        segment_sheet.write_row(
            header_row - 1,
            0,
            headers,
            table_header_format
        )


        for row_offset, (_, row) in enumerate(
                dataframe.iterrows()
        ):

            excel_row = (
                first_data_row
                + row_offset
            )


            #Write source values produced by Python
            segment_sheet.write(
                excel_row - 1,
                0,
                row.iloc[0],
                body_text
            )

            segment_sheet.write(
                excel_row - 1,
                1,
                row["Policies"],
                integer_format
            )

            segment_sheet.write(
                excel_row - 1,
                2,
                row["Exposure"],
                decimal_2_format
            )

            segment_sheet.write(
                excel_row - 1,
                3,
                row["ExpectedClaims"],
                decimal_2_format
            )

            segment_sheet.write(
                excel_row - 1,
                4,
                row["ExpectedLosses"],
                currency_format
            )


            #Excel calculates the derived pricing metrics
            segment_sheet.write_formula(
                excel_row - 1,
                5,
                f"=D{excel_row}/C{excel_row}",
                decimal_4_format
            )

            segment_sheet.write_formula(
                excel_row - 1,
                6,
                f"=E{excel_row}/D{excel_row}",
                currency_format
            )

            segment_sheet.write_formula(
                excel_row - 1,
                7,
                f"=E{excel_row}/C{excel_row}",
                currency_format
            )

            segment_sheet.write_formula(
                excel_row - 1,
                8,
                f"=H{excel_row}/'Model Summary'!$B$9",
                relativity_format
            )


        last_data_row = (
            first_data_row
            + len(dataframe)
            - 1
        )


        #Same green-yellow-red relativity color scale as the reference workbook
        segment_sheet.conditional_format(
            f"I{first_data_row}:I{last_data_row}",
            {
                "type": "3_color_scale",
                "min_color": green_cf,
                "mid_color": yellow_cf,
                "max_color": red_cf
            }
        )


    write_segment_table(
        driver_age,
        header_row=3,
        first_data_row=4,
        label_header="Driver Age Group"
    )


    write_segment_table(
        vehicle_age,
        header_row=13,
        first_data_row=14,
        label_header="Vehicle Age Group"
    )


    write_segment_table(
        bonus_malus,
        header_row=22,
        first_data_row=23,
        label_header="Bonus-Malus Group"
    )


    write_segment_table(
        density,
        header_row=31,
        first_data_row=32,
        label_header="Density Group"
    )


    segment_sheet.freeze_panes(
        3,
        0
    )


    #=====================================================
    #STEP 8: VALIDATION SHEET
    #=====================================================

    validation_sheet = workbook.add_worksheet(
        "Validation"
    )


    validation_sheet.set_column(
        "A:A",
        12
    )

    validation_sheet.set_column(
        "B:J",
        16
    )


    validation_sheet.merge_range(
        "A1:J1",
        "Holdout Validation by Predicted Risk Decile",
        title_format
    )


    validation_headers = [
        "Risk Decile",
        "Policies",
        "Exposure",
        "Actual Claims",
        "Actual Losses",
        "Predicted Claims",
        "Predicted Losses",
        "Actual Pure Premium",
        "Predicted Pure Premium",
        "Actual / Expected"
    ]


    validation_sheet.write_row(
        "A3",
        validation_headers,
        table_header_format
    )


    for row_offset, (_, row) in enumerate(
            validation_deciles.iterrows()
    ):

        excel_row = (
            4
            + row_offset
        )


        validation_sheet.write(
            excel_row - 1,
            0,
            row["RiskDecile"],
            integer_format
        )

        validation_sheet.write(
            excel_row - 1,
            1,
            row["Policies"],
            integer_format
        )

        validation_sheet.write(
            excel_row - 1,
            2,
            row["Exposure"],
            decimal_2_format
        )

        validation_sheet.write(
            excel_row - 1,
            3,
            row["ActualClaims"],
            integer_format
        )

        validation_sheet.write(
            excel_row - 1,
            4,
            row["ActualLosses"],
            currency_format
        )

        validation_sheet.write(
            excel_row - 1,
            5,
            row["PredictedClaims"],
            decimal_2_format
        )

        validation_sheet.write(
            excel_row - 1,
            6,
            row["PredictedLosses"],
            currency_format
        )


        #Excel calculates the validation metrics
        validation_sheet.write_formula(
            excel_row - 1,
            7,
            f"=E{excel_row}/C{excel_row}",
            currency_format
        )

        validation_sheet.write_formula(
            excel_row - 1,
            8,
            f"=G{excel_row}/C{excel_row}",
            currency_format
        )

        validation_sheet.write_formula(
            excel_row - 1,
            9,
            f"=E{excel_row}/G{excel_row}",
            relativity_format
        )


    validation_sheet.conditional_format(
        "J4:J13",
        {
            "type": "3_color_scale",
            "min_color": green_cf,
            "mid_color": yellow_cf,
            "max_color": red_cf
        }
    )


    validation_sheet.freeze_panes(
        3,
        0
    )


    #=====================================================
    #STEP 9: DASHBOARD SHEET
    #=====================================================

    dashboard = workbook.add_worksheet(
        "Dashboard"
    )


    #Match the reference workbook's column widths
    dashboard.set_column(
        "A:A",
        14
    )

    dashboard.set_column(
        "B:C",
        12
    )

    dashboard.set_column(
        "D:D",
        14
    )

    dashboard.set_column(
        "E:F",
        12
    )

    dashboard.set_column(
        "G:G",
        16
    )

    dashboard.set_column(
        "H:I",
        12
    )

    dashboard.set_column(
        "J:J",
        18
    )

    dashboard.set_column(
        "K:L",
        12
    )


    #Hide worksheet gridlines for the dashboard only
    dashboard.hide_gridlines(
        2
    )


    #-----------------------------------------------------
    #Dashboard Title
    #-----------------------------------------------------

    dashboard.merge_range(
        "A1:L2",
        "Automobile Insurance Pricing Analysis Dashboard",
        dashboard_title_format
    )


    #-----------------------------------------------------
    #Full Portfolio Model Section
    #-----------------------------------------------------

    dashboard.merge_range(
        "A4:L4",
        "Full Portfolio Model",
        section_blue_format
    )


    #Labels are centered across three-column KPI blocks
    dashboard.write(
        "A5",
        "Policies",
        dashboard_label_format
    )

    dashboard.write(
        "D5",
        "Exposure",
        dashboard_label_format
    )

    dashboard.write(
        "G5",
        "Predicted Frequency",
        dashboard_label_format
    )

    dashboard.write(
        "J5",
        "Predicted Pure Premium",
        dashboard_label_format
    )


    #Fill the rest of each label block with the same format
    dashboard.set_row(
        4,
        None
    )

    dashboard.write_blank(
        "B5",
        None,
        dashboard_label_format
    )

    dashboard.write_blank(
        "C5",
        None,
        dashboard_label_format
    )

    dashboard.write_blank(
        "E5",
        None,
        dashboard_label_format
    )

    dashboard.write_blank(
        "F5",
        None,
        dashboard_label_format
    )

    dashboard.write_blank(
        "H5",
        None,
        dashboard_label_format
    )

    dashboard.write_blank(
        "I5",
        None,
        dashboard_label_format
    )

    dashboard.write_blank(
        "K5",
        None,
        dashboard_label_format
    )

    dashboard.write_blank(
        "L5",
        None,
        dashboard_label_format
    )


    #KPI values
    dashboard.merge_range(
        "A6:C8",
        "",
        portfolio_kpi_integer
    )

    dashboard.write_formula(
        "A6",
        "='Model Summary'!B4",
        portfolio_kpi_integer
    )


    dashboard.merge_range(
        "D6:F8",
        "",
        portfolio_kpi_decimal
    )

    dashboard.write_formula(
        "D6",
        "='Model Summary'!B5",
        portfolio_kpi_decimal
    )


    dashboard.merge_range(
        "G6:I8",
        "",
        portfolio_kpi_percent
    )

    dashboard.write_formula(
        "G6",
        "='Model Summary'!B7",
        portfolio_kpi_percent
    )


    dashboard.merge_range(
        "J6:L8",
        "",
        portfolio_kpi_currency
    )

    dashboard.write_formula(
        "J6",
        "='Model Summary'!B9",
        portfolio_kpi_currency
    )


    #-----------------------------------------------------
    #Holdout Validation Section
    #-----------------------------------------------------

    dashboard.merge_range(
        "A10:L10",
        "Holdout Validation",
        section_orange_format
    )


    dashboard.write(
        "A11",
        "Actual Pure Premium",
        dashboard_label_format
    )

    dashboard.write(
        "D11",
        "Predicted Pure Premium",
        dashboard_label_format
    )

    dashboard.write(
        "G11",
        "Loss Prediction Error",
        dashboard_label_format
    )

    dashboard.write(
        "J11",
        "Ordered Gini",
        dashboard_label_format
    )


    for cell in [
        "B11",
        "C11",
        "E11",
        "F11",
        "H11",
        "I11",
        "K11",
        "L11"
    ]:

        dashboard.write_blank(
            cell,
            None,
            dashboard_label_format
        )


    dashboard.merge_range(
        "A12:C14",
        "",
        validation_kpi_currency
    )

    dashboard.write_formula(
        "A12",
        "='Model Summary'!B12",
        validation_kpi_currency
    )


    dashboard.merge_range(
        "D12:F14",
        "",
        validation_kpi_currency
    )

    dashboard.write_formula(
        "D12",
        "='Model Summary'!B13",
        validation_kpi_currency
    )


    dashboard.merge_range(
        "G12:I14",
        "",
        validation_kpi_percent
    )

    dashboard.write_formula(
        "G12",
        "='Model Summary'!B14",
        validation_kpi_percent
    )


    dashboard.merge_range(
        "J12:L14",
        "",
        validation_kpi_decimal
    )

    dashboard.write_formula(
        "J12",
        "='Model Summary'!B18",
        validation_kpi_decimal
    )


    #-----------------------------------------------------
    #Validation Note
    #-----------------------------------------------------

    dashboard.merge_range(
        "A16:L17",
        "",
        note_format
    )


    dashboard.write_formula(
        "A16",
        '="Validation note: the combined model improved Tweedie deviance by "'
        '&TEXT(\'Model Summary\'!B17,"0.00%")'
        '&" versus the baseline and produced a positive ordered Gini of "'
        '&TEXT(\'Model Summary\'!B18,"0.0000")'
        '&". The model ranked high-risk policies effectively, but underpredicted "'
        '&"total holdout losses by "'
        '&TEXT(ABS(\'Model Summary\'!B14),"0.00%")'
        '&", indicating room for further calibration."',
        note_format
    )


    #=====================================================
    #STEP 10: DASHBOARD CHARTS
    #=====================================================

    #-----------------------------------------------------
    #Driver Age Pure Premium
    #-----------------------------------------------------

    age_chart = workbook.add_chart(
        {
            "type": "column"
        }
    )


    age_chart.add_series(
        {
            "name": "Predicted Pure Premium",
            "categories": "='Segment Pricing'!$A$4:$A$10",
            "values": "='Segment Pricing'!$H$4:$H$10",
            "fill": {
                "color": chart_blue
            },
            "border": {
                "none": True
            }
        }
    )


    age_chart.set_style(
        2
    )

    age_chart.set_title(
        {
            "none": True
        }
    )

    age_chart.set_legend(
        {
            "position": "bottom"
        }
    )

    age_chart.set_x_axis(
        {
            "major_tick_mark": "none",
            "minor_tick_mark": "none"
        }
    )

    age_chart.set_y_axis(
        {
            "num_format": r"\€#,##0.00",
            "major_gridlines": {
                "visible": True,
                "line": {
                    "color": grid_gray,
                    "dash_type": "dash"
                }
            },
            "major_tick_mark": "none",
            "minor_tick_mark": "none"
        }
    )

    age_chart.set_chartarea(
        {
            "border": {
                "color": chart_border
            }
        }
    )

    age_chart.set_plotarea(
        {
            "border": {
                "none": True
            }
        }
    )

    age_chart.set_size(
        {
            "width": 620,
            "height": 320
        }
    )


    dashboard.insert_chart(
        "A19",
        age_chart
    )


    #-----------------------------------------------------
    #Bonus-Malus Pure Premium
    #-----------------------------------------------------

    bonus_chart = workbook.add_chart(
        {
            "type": "column"
        }
    )


    bonus_chart.add_series(
        {
            "name": "Predicted Pure Premium",
            "categories": "='Segment Pricing'!$A$23:$A$28",
            "values": "='Segment Pricing'!$H$23:$H$28",
            "fill": {
                "color": chart_blue
            },
            "border": {
                "none": True
            }
        }
    )


    bonus_chart.set_style(
        2
    )

    bonus_chart.set_title(
        {
            "none": True
        }
    )

    bonus_chart.set_legend(
        {
            "position": "bottom"
        }
    )

    bonus_chart.set_x_axis(
        {
            "major_tick_mark": "none",
            "minor_tick_mark": "none"
        }
    )

    bonus_chart.set_y_axis(
        {
            "num_format": r"\€#,##0.00",
            "major_gridlines": {
                "visible": True,
                "line": {
                    "color": grid_gray,
                    "dash_type": "dash"
                }
            },
            "major_tick_mark": "none",
            "minor_tick_mark": "none"
        }
    )

    bonus_chart.set_chartarea(
        {
            "border": {
                "color": chart_border
            }
        }
    )

    bonus_chart.set_plotarea(
        {
            "border": {
                "none": True
            }
        }
    )

    bonus_chart.set_size(
        {
            "width": 620,
            "height": 320
        }
    )


    dashboard.insert_chart(
        "G19",
        bonus_chart,
        {
            "x_offset": 60
        }
    )


    #-----------------------------------------------------
    #Geographic Density Pure Premium
    #-----------------------------------------------------

    density_chart = workbook.add_chart(
        {
            "type": "column"
        }
    )


    density_chart.add_series(
        {
            "name": "Predicted Pure Premium",
            "categories": "='Segment Pricing'!$A$32:$A$36",
            "values": "='Segment Pricing'!$H$32:$H$36",
            "fill": {
                "color": chart_blue
            },
            "border": {
                "none": True
            }
        }
    )


    density_chart.set_style(
        2
    )

    density_chart.set_title(
        {
            "none": True
        }
    )

    density_chart.set_legend(
        {
            "position": "bottom"
        }
    )

    density_chart.set_x_axis(
        {
            "major_tick_mark": "none",
            "minor_tick_mark": "none"
        }
    )

    density_chart.set_y_axis(
        {
            "num_format": r"\€#,##0.00",
            "major_gridlines": {
                "visible": True,
                "line": {
                    "color": grid_gray,
                    "dash_type": "dash"
                }
            },
            "major_tick_mark": "none",
            "minor_tick_mark": "none"
        }
    )

    density_chart.set_chartarea(
        {
            "border": {
                "color": chart_border
            }
        }
    )

    density_chart.set_plotarea(
        {
            "border": {
                "none": True
            }
        }
    )

    density_chart.set_size(
        {
            "width": 620,
            "height": 320
        }
    )


    dashboard.insert_chart(
        "A36",
        density_chart
    )


    #-----------------------------------------------------
    #Actual vs Predicted Pure Premium by Risk Decile
    #-----------------------------------------------------

    validation_chart = workbook.add_chart(
        {
            "type": "line"
        }
    )


    validation_chart.add_series(
        {
            "name": "Actual",
            "categories": "=Validation!$A$4:$A$13",
            "values": "=Validation!$H$4:$H$13",
            "line": {
                "color": chart_blue,
                "width": 2
            },
            "marker": {
                "type": "none"
            },
            "smooth": True
        }
    )


    validation_chart.add_series(
        {
            "name": "Predicted",
            "categories": "=Validation!$A$4:$A$13",
            "values": "=Validation!$I$4:$I$13",
            "line": {
                "color": chart_orange,
                "width": 2
            },
            "marker": {
                "type": "none"
            },
            "smooth": True
        }
    )


    validation_chart.set_style(
        2
    )

    validation_chart.set_title(
        {
            "none": True
        }
    )

    validation_chart.set_legend(
        {
            "position": "bottom"
        }
    )

    validation_chart.set_x_axis(
        {
            "num_format": "0",
            "major_tick_mark": "none",
            "minor_tick_mark": "none"
        }
    )

    validation_chart.set_y_axis(
        {
            "num_format": r"\€#,##0.00",
            "major_gridlines": {
                "visible": True,
                "line": {
                    "color": grid_gray,
                    "dash_type": "dash"
                }
            },
            "major_tick_mark": "none",
            "minor_tick_mark": "none"
        }
    )

    validation_chart.set_chartarea(
        {
            "border": {
                "color": chart_border
            }
        }
    )

    validation_chart.set_plotarea(
        {
            "border": {
                "none": True
            }
        }
    )

    validation_chart.set_size(
        {
            "width": 620,
            "height": 320
        }
    )


    dashboard.insert_chart(
        "G36",
        validation_chart,
        {
            "x_offset": 60
        }
    )


    #=====================================================
    #STEP 11: METHODOLOGY SHEET
    #=====================================================

    methodology_sheet = workbook.add_worksheet(
        "Methodology"
    )


    methodology_sheet.set_column(
        "A:A",
        22
    )

    methodology_sheet.set_column(
        "B:B",
        80
    )

    methodology_sheet.set_column(
        "C:C",
        45
    )

    methodology_sheet.set_column(
        "D:F",
        10
    )


    methodology_sheet.merge_range(
        "A1:C1",
        "Project Methodology and Limitations",
        title_format
    )


    methodology_sheet.write_row(
        "A3",
        [
            "Section",
            "Details",
            ""
        ],
        table_header_format
    )


    methodology_rows = [
        [
            "Business objective",
            "Estimate automobile insurance expected claim cost by separating claim frequency and claim severity, then combining the predictions into policy-level pure premium.",
            ""
        ],
        [
            "Frequency model",
            "Poisson GLM using policy exposure as observation weight. Rating variables include driver age, vehicle age, bonus-malus, area, brand, fuel type, region, vehicle power, and log population density.",
            ""
        ],
        [
            "Severity model",
            "Gamma GLM using claim count as observation weight. The model predicts average claim severity for claim-bearing policies.",
            ""
        ],
        [
            "Pure premium",
            "Predicted Frequency × Predicted Severity. Segment pricing relativities compare modeled pure premium with the overall modeled portfolio pure premium.",
            ""
        ],
        [
            "Validation",
            "20% holdout sample on a reconciled frequency/severity dataset. Performance reviewed using total-loss calibration, Tweedie deviance, risk deciles, actual-to-expected ratios, and ordered Gini.",
            ""
        ],
        [
            "Key validation result",
            (
                f"Model Tweedie deviance improved "
                f"{((baseline_tweedie_deviance-model_tweedie_deviance)/baseline_tweedie_deviance):.2%} "
                f"over baseline and ordered Gini was {ordered_gini:.4f}. "
                f"Holdout losses were underpredicted by "
                f"{abs((validation_predicted_losses-validation_actual_losses)/validation_actual_losses):.2%}, "
                f"with the largest underprediction concentrated in high-risk deciles."
            ),
            ""
        ],
        [
            "Limitation",
            "The public frequency and severity source files do not reconcile perfectly. Combined validation therefore uses only claims with corresponding positive observed claim amounts.",
            ""
        ],
        [
            "Limitation",
            "Claim counts, exposure, and claim amounts were capped for modeling to limit the influence of extreme records. Results are suitable for portfolio demonstration, not production ratemaking.",
            ""
        ],
        [
            "Source",
            "freMTPL2 motor insurance data via OpenML datasets 41214 and 41215.",
            "https://www.openml.org"
        ],
        [
            "Reference",
            "Scikit-learn insurance pricing example using Poisson, Gamma, and Tweedie models.",
            "https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html"
        ]
    ]


    for row_number, row_data in enumerate(
            methodology_rows,
            start=4
    ):

        methodology_sheet.write(
            row_number - 1,
            0,
            row_data[0],
            body_text
        )

        methodology_sheet.write(
            row_number - 1,
            1,
            row_data[1],
            body_text
        )

        methodology_sheet.write(
            row_number - 1,
            2,
            row_data[2],
            body_text
        )


    methodology_sheet.freeze_panes(
        3,
        0
    )


    #=====================================================
    #STEP 12: SET WORKSHEET ORDER
    #=====================================================

    #Sheets were deliberately created in the same order
    #as the reference workbook:
    #Model Summary
    #Segment Pricing
    #Validation
    #Dashboard
    #Methodology


print("\n======================================")
print("EXCEL DASHBOARD COMPLETE")
print("======================================")

print(
    f"Workbook saved to: "
    f"{output_file}"
)

print(
    "\nThis workbook matches the reference dashboard layout, "
    "charts, color palette, tables, and methodology structure."
)

