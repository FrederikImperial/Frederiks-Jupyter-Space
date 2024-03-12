# %%
# %%
#change working directory to same as file
change_work_directory = True

if change_work_directory:

    import os

    # Absoluten Pfad der Python-Datei erhalten
    script_path = os.path.abspath(__file__)

    # Verzeichnis extrahieren
    script_dir = os.path.dirname(script_path)

    # Arbeitsverzeichnis ändern
    os.chdir(script_dir)


# %%
Strake_ID_input = 115
thickness_input = 19.05

print('Strake_ID_input = ',Strake_ID_input)

# %%
#Einlesen von thickness und Strake_ID

# Dateipfad zur Textdatei
file_path = "../data/input/input_Strake_ID.txt"

# Variablen initialisieren
Strake_ID_input = None
thickness_input = None

# Versuchen, die Werte aus der Textdatei zu lesen
try:
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('=')
            if len(parts) == 2:
                var_name = parts[0].strip()
                var_value = parts[1].strip()
                if var_name == 'Strake_ID_input':
                    Strake_ID_input = float(var_value)
                elif var_name == 'thickness_input':
                    thickness_input = float(var_value)

    # Überprüfen, ob alle Variablen erfolgreich gesetzt wurden
    if Strake_ID_input is None or thickness_input is None:
        raise ValueError("Fehler beim Lesen der Werte aus der Datei.")

    # Ausgabe der Variablenwerte (optional)
    print("Strake_ID_input:", Strake_ID_input)
    print("thickness_input:", thickness_input)

except FileNotFoundError:
    print("Die Datei wurde nicht gefunden.")
except ValueError as e:
    print("Fehler:", e)


# %%
import pandas as pd

# Laden des DataFrames aus der CSV-Datei und Festlegen von 'index' als Index
loaded_df = pd.read_csv('../data/temporary/membrane_stress_resultants_for_lc.csv', index_col='index')

print('dataframe df, containing stress resultants of the whole tower for the selected load case, loaded from ../data/temporary/membrane_stress_resultants_for_lc.csv')


# %%
create_Subset = True

if create_Subset:

    #Create Subset of dataframe to select strake

    # Subset des DataFrame basierend auf der Bedingung für Strake-ID
    subset_df = loaded_df[(loaded_df['Strake_ID'] >= Strake_ID_input) & (loaded_df['Strake_ID'] <= Strake_ID_input)]


# %%
#Calculate Strake Height

# Voraussetzung: subset_df ist bereits definiert und enthält den gewünschten Teil des DataFrames

# Berechnung der Differenz zwischen dem größten und kleinsten Wert in der Spalte 'z' um L_strake zu erhalten
L_strake = subset_df['z'].max() - subset_df['z'].min() + 1

print('L_strake = ',L_strake)

r = subset_df.iloc[0]['r']
print('r_value = ',r)


# Annahme: subset_df ist bereits definiert und enthält die Spalten 'N_z_total', 'theta' und 'z'

# Index der Zeile mit dem maximalen Wert von 'N_z_total' in subset_df
min_index = subset_df['N_z_total'].idxmin()

# Theta und z aus der gleichen Zeile wie N_z_total max auslesen
theta_value = subset_df.at[min_index, 'theta']
z_value = subset_df.at[min_index, 'z']

# Maximalen Wert von N_z_total auslesen
min_N_z = subset_df.at[min_index, 'N_z_total']

# Ausgabe der Werte
print("Minimaler Wert 'N_z_min':", min_N_z)
print("Zugehöriger Wert 'theta':", theta_value)
print("Zugehöriger Wert 'z':", z_value)


# %%
t=thickness_input


import math

#relative length parameters
# Calculate omega
omega =L_strake / math.sqrt(r * t)

# Calculate Omega
Omega_2 = (t / r) * omega

print("omega =", omega)
print("Omega_2 =", Omega_2)

#length domains
#assume medium length cylinder. check later

C_x = 1

#take L for single strake and assume LC1 or LC2 at both edges. Further gloabal buckling checks required!
E = 200000

# Berechnung von sigma_x_Rcr
sigma_x_Rcr = 0.605 * E * C_x * (t / r)

print("sigma_x_Rcr =", sigma_x_Rcr, "[N/mm²]")

# Given parameters
lambda_x0_bar = 0.1
alpha_xG = 0.83
Q_x = 40.0 #Change to 40 for Execution controll class A


# Calculate alpha_xI
delta_0_over_t = (1 / Q_x) * math.sqrt(r/t)
alpha_xI = 1 / (1 + 2.2 * delta_0_over_t ** 0.75)

# Calculate alpha_x
alpha_x = alpha_xG * alpha_xI

print("alpha_xI", alpha_xI)
print("delta_0_over_t =", delta_0_over_t)
print("alpha_x =", alpha_x)

# Given parameters
f_yk = 355  # in N/mm^2

#plastic range factor
beta_x = 1 - 0.75 / (1 + 1.1 * delta_0_over_t)

#interaction exponent
eta_x0 = 1.35 - 0.10 * delta_0_over_t
eta_xp = 1 / (0.45 + 0.72 * delta_0_over_t)

#from Cl. 9.5.2 (3),(5)
lambda_x_bar = math.sqrt(f_yk / sigma_x_Rcr)
lambda_xp_bar = math.sqrt(alpha_x / (1 - beta_x))


eta_x = ((lambda_x_bar * (eta_xp - eta_x0) + lambda_xp_bar * eta_x0 - lambda_x0_bar * eta_xp) / (lambda_xp_bar - lambda_x0_bar))
chi_xh = 1.10
chi_h = 1.10
chi = 1

# Calculation of chi based on lambda values
if lambda_x_bar <= lambda_x0_bar:
    chi = chi_h - (lambda_x_bar / lambda_x0_bar) * (chi_h - 1)
elif lambda_x0_bar < lambda_x_bar < lambda_xp_bar:
    chi = 1 - beta_x * ((lambda_x_bar - lambda_x0_bar) / (lambda_xp_bar - lambda_x0_bar)) ** eta_x
else:
    chi = alpha_x / lambda_x_bar ** 2

sigma_x_Rk = chi * f_yk
sigma_x_Rd = sigma_x_Rk / 1.1

# Check if sigma_x_Rd * t exceeds the limit
sigma_x_Rd_t = sigma_x_Rd * t

Utilization_output = -min_N_z /sigma_x_Rd_t

# Runden auf 2 Nachkommastellen
Utilization_output= round(Utilization_output, 2)


print("Plastic range factor (beta_x):", beta_x)
print("Interaction exponent (eta_x0):", eta_x0)
print("Interaction exponent (eta_xp):", eta_xp)
print("Lambda_x_bar:", lambda_x_bar)
print("Lambda_xp_bar:", lambda_xp_bar)
print("Eta_x:", eta_x)
print("Chi:", chi)
print("Sigma_x_Rk:", sigma_x_Rk)
print("Sigma_x_Rd:", sigma_x_Rd)
print("Sigma_x_Rd * t:", sigma_x_Rd_t, " > N_z = ", -min_N_z, " ?")

if sigma_x_Rd_t > -min_N_z:
    print("OK!")
else:
    print("NOT OK!")


# %%
# Dateipfad zur Ausgabedatei
file_path = "../data/output/output_Utilization.txt"

# Versuchen, den Wert in die Datei zu schreiben
try:
    with open(file_path, 'w') as file:
        file.write(f"Utilization = {Utilization_output}\n")
    print("Der Wert wurde erfolgreich in die Datei geschrieben.")
except IOError:
    print("Fehler beim Schreiben in die Datei.")


