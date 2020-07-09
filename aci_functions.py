import numpy as np
import pandas as pd


# dict containing correction factors for each species
# order of factors for "c" and "DHa" is Kc (Pa) Ko (kPa) SC/O VcMax j TPU Rd gm
# order of factors for "DHd" and "DS" is TPU gm
species_vals = {"tobbaco": {"c" : [35.9774, 
                                   12.3772, 
                                   -1.8730, 
                                   26.355,
                                   17.71,
                                   25.47,
                                   18.715,
                                   20.01],
                            "DHa" : [80.99, 
                                     23.72, 
                                     -24.4754,
                                    65.33,
                                     43.9,
                                     62.99,
                                     46.39,
                                     49.6],
                            "DHd": [182.14,
                                    437.4],
                            "DS":  [0.588,
                                    1.4]},
                "arabidopsis": {"c": [35.9774,
                                      12.3772,
                                      0.2306,
                                      26.355,
                                      17.71,
                                      25.47,
                                      18.715,
                                      20.01],
                                "DHa": [80.99,
                                        23.72,
                                        -18.671,
                                        65.33,
                                        43.9,
                                        62.99,
                                        46.39,
                                        49.6],
                               "DHd": [182.14,
                                        437.4],
                                "DS":  [0.588,
                                        1.4]}}


def read_licor(file):
    """Return pandas df from licor output in excel or tsv format.
    
    Keyword arguments:
    file -- path to licor data file
    """
    import pandas as pd
    import openpyxl as op
    import csv
    col_values = {"a", "ca", "pci", "qin"}
    rows_to_skip = None
    try:
        wb = op.load_workbook(file)
        fd = wb.worksheets[0]
        for (i, row) in enumerate(fd.rows):
            values = [row[j].value.lower()  if isinstance(row[j].value, str) is True else row[j].value for j in range(len(row))]
            if col_values.issubset(set(values)):
                rows_to_skip = i+2
                header = values
                break
                
        df = pd.read_excel(file, skiprows= rows_to_skip, header= None)
        df.columns = header
        
            
    except op.reader.excel.InvalidFileException:
        with open(file, "r") as tsvfile:
            rd = csv.reader(tsvfile, delimiter="\t" )
            for (i, row) in enumerate(rd):
                values = [row[j].lower()  if isinstance(row[j], str) is True else row[j] for j in range(len(row))]
                if col_values.issubset(set(values)):
                    rows_to_skip = i+2
                    header = values
                    break
        df = pd.read_table(file, skiprows=rows_to_skip, header=None)
        df.columns = header
    return df
    

def split_light_curve(df):
    """Return a df with a label for each unique integer light level in the qin column.
    
    Keyword arguments:
    df -- dataframe output by the read_licor function.
    """
    import pandas as pd
    light_label = [round(v) for v in df["qin"]]
    df["light_label"] = light_label
    return df
    

def get_plt_df(df):
    """Return df subset with uneeded columns and rows outside of CO2 series removed.
    
    Keyword arguments:
    df -- dataframe returned by split_light_curve function
    """
    import pandas as pd
    plt_df = df[["a","pci","ci","ca","light_label"]]
    index_to_drop  = []
    for i in set(plt_df["light_label"]):
        df_sub = plt_df.loc[plt_df["light_label"] == i]
        for j in df_sub.index:
            if j - 1 > min(df_sub.index):
                if (df_sub.loc[j]["ca"] - df.loc[j-1]["ca"]) < 0:
                    index_to_drop.append(j)
            else:
                if (df_sub.loc[j+1]["ca"] - df.loc[j]["ca"]) < 0:
                    index_to_drop.append(j)
    plt_df = plt_df.drop(index_to_drop)
    return(plt_df)
                
                
def plot_curve(df):
    """Plots raw data of A/Ci curve for each level of PAR.
    
    Keyword arguments:
    df -- datframe returned by get_plt_df function.
    """
    import pandas as pd
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    #ax.scatter(df["pci"], df["a"], c = df["light_label"])
    for light_level in set(df["light_label"]):
        df_sub = df.loc[df["light_label"] == light_level]
        ax.scatter(df_sub["pci"], df_sub["a"], label = light_level)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip([int(i) for i in labels], handles), key=lambda t: t[0]))
    plt.ylabel("A")
    plt.xlabel("pCi")
    plt.legend(labels=labels, handles = handles, bbox_to_anchor = (1.25,1), title = "PAR")
    plt.tight_layout()
 
 
 # Functions for converting paramaters to 25C equivalent
def Kc_func(t_leaf, species):
    """Return temperature adjusted  equilibrium constant for CO2.
    
    Keyword Arguments:
    
    t_leaf -- leaf temperature in celcius 
    species -- name of species to use for adjustment (must be in species_vals.keys())
    """
    return np.exp(species_vals[species]["c"][0]-
                  (species_vals[species]["DHa"][0] / (0.008314*(273.15 + t_leaf))))

def Ko_func(t_leaf, species):
    """Return temperature adjusted  equilibrium constant for O2.
    
    Keyword Arguments:
    
    t_leaf -- leaf temperature in celcius 
    species -- name of species to use for adjustment (must be in species_vals.keys())
    """
    return np.exp(species_vals[species]["c"][1]-
                  (species_vals[species]["DHa"][1] / (0.008314*(273.15 + t_leaf))))

def SC_O_func(t_leaf, species):
    """Return temperature adjusted SC / O value.
    
    Keyword Arguments:
    
    t_leaf -- leaf temperature in celcius 
    species -- name of species to use for adjustment (must be in species_vals.keys())
    """
    return np.exp(species_vals[species]["c"][2]-
                  (species_vals[species]["DHa"][2] / (0.008314*(273.15 + t_leaf))))

def gammastar_func(t_leaf, O2, species, aG):
    """Return temperature adjusted gammastar value.
    
    Keyword Arguments:
    
    t_leaf -- leaf temperature in celcius 
    O2 -- partial pressure of O2
    species -- name of species to use for adjustment (must be in species_vals.keys())
    aG -- alpha g value
    """
    return (1-aG) * 0.5 * O2 *1000 / SC_O_func(t_leaf, species)

def Vcmax_func(t_leaf, species):
    """Return temperature adjusted rubisco Vcmax.
    
    Keyword Arguments:
    
    t_leaf -- leaf temperature in celcius 
    species -- name of species to use for adjustment (must be in species_vals.keys())
    """
    return np.exp(species_vals[species]["c"][3]-
                  (species_vals[species]["DHa"][3] / (0.008314*(273.15 + t_leaf))))
def J_func(t_leaf, species):
    """Return temperature adjusted linear electron flow.
    
    Keyword Arguments:
    
    t_leaf -- leaf temperature in celcius 
    species -- name of species to use for adjustment (must be in species_vals.keys())
    """
    return np.exp(species_vals[species]["c"][4]-
                  (species_vals[species]["DHa"][4] / (0.008314*(273.15 + t_leaf))))

def TPU_func(t_leaf, species):
    """Return temperature adjusted triose phosphate usage.
    
    Keyword Arguments:
    
    t_leaf -- leaf temperature in celcius 
    species -- name of species to use for adjustment (must be in species_vals.keys())
    """
    return (np.exp(species_vals[species]["c"][5]-
                  (species_vals[species]["DHa"][5] / (0.008314*(273.15 + t_leaf)))))/(1
                    +np.exp((species_vals[species]["DS"][0]*(t_leaf +273.15))-
                      species_vals[species]["DHd"][0])/(0.008314*(t_leaf+273.15)))

def Rd_func(t_leaf, species):
    """Return temperature adjusted Rd value.
    
    Keyword Arguments:
    
    t_leaf -- leaf temperature in celcius 
    species -- name of species to use for adjustment (must be in species_vals.keys())
    """
    return np.exp(species_vals[species]["c"][6]-
                  (species_vals[species]["DHa"][6] / (0.008314*(273.15 + t_leaf))))

def gm_func(t_leaf, species):
    """Return temperature adjusted mesophyll conductance.
    
    Keyword Arguments:
    
    t_leaf -- leaf temperature in celcius 
    species -- name of species to use for adjustment (must be in species_vals.keys())
    """
    return (np.exp(species_vals[species]["c"][7]-
                  (species_vals[species]["DHa"][7] / (0.008314*(273.15 + t_leaf)))))/(1
                    +np.exp((species_vals[species]["DS"][1]*(t_leaf +273.15))-
                      species_vals[species]["DHd"][1])/(0.008314*(t_leaf+273.15)))


# Fitting functions

# 
def CcFunc( df, gm ):
    """ Calculates the partial pressure of CO2 at the chloroplast.
    
    Keyword Arguments:
    df -- dataframe contaiing the columns 'pci' and 'a'
    gm -- mesophyl conductance
    """
    return df['pci'] - df['a']/gm
    
# 
def CoefFunc( aG , gammastar, Cc ):
    """ Returns fitting coeficcent used for all functions.
    
    Keyword Arguments:
    aG -- alpha G value
    gammastar -- gammastar value
    Cc -- partial pressure of Co2 at chloroplast 
    """
    return 1-((1-aG)*gammastar/Cc)

# 
def AcFunc(df, aG, aS, Rd, Vcmax, j, TPU, gm):
    """Assimilation assuming Rubisco limitation.
    
    Keyword Arguments:
    df -- dataframe with columns 'pci' and 'a'
    aG -- alpha G value
    aS -- alpha S value
    Rd -- Rd value
    Vcmax -- rubisco Vcmax value
    j -- linear electron flow
    TPU -- Triose Phosphate Usage
    gm -- mesophyl conductance 
    
    """
    Cc = CcFunc( df, gm )
    coef = CoefFunc( aG , gammastar, Cc )
    return coef * (Vcmax *Cc)/((Cc +Kc * (1 + O/Ko)))-Rd

# 
def AjFunc(df, aG, aS, Rd, Vcmax, j, TPU, gm):
    """ Assimilation assuming RUBP regeneration limitation.
    
    Keyword Arguments:
    df -- dataframe with columns 'pci' and 'a'
    aG -- alpha G value
    aS -- alpha S value
    Rd -- Rd value
    Vcmax -- rubisco Vcmax value
    j -- linear electron flow
    TPU -- Triose Phosphate Usage
    gm -- mesophyl conductance 
    """
    Cc = CcFunc( df, gm )
    coef = CoefFunc( aG , gammastar, Cc )
    return coef * j/(4+(4+8*aG+4*aS)*2*((1-aG)*gammastar)/Cc)-Rd # fit a

# Assimilation assuming TPU limitation     
def ApFunc(df, aG, aS, Rd, Vcmax, j, TPU, gm):
    """Assimilation assuming TPU limitation.
    
    Keyword Arguments:
    df -- dataframe with columns 'pci' and 'a'
    aG -- alpha G value
    aS -- alpha S value
    Rd -- Rd value
    Vcmax -- rubisco Vcmax value
    j -- linear electron flow
    TPU -- Triose Phosphate Usage
    gm -- mesophyl conductance 
    """
    Cc = CcFunc( df, gm )
    coef = CoefFunc( aG , gammastar, Cc )
    result = coef * (3*TPU/(1-0.5*(1+3*aG+4*aS)*2*(1-aG)*gammastar/Cc))-Rd
    
    # insert infinite y-values where the x-value (Cc) is low
    result[np.where( Cc < 20 )[0]] = np.inf
    
    return result
# 
def Afunc(df, aG, aS, Rd, Vcmax, j, TPU, gm):
    """Combined function, returns the minimum of the three functions at each x.
    
    Keyword Arguments:
    df -- dataframe with columns 'pci' and 'a'
    aG -- alpha G value
    aS -- alpha S value
    Rd -- Rd value
    Vcmax -- rubisco Vcmax value
    j -- linear electron flow
    TPU -- Triose Phosphate Usage
    gm -- mesophyl conductance
    """
    ac = AcFunc(df, aG, aS, Rd, Vcmax, j, TPU, gm)
    aj = AjFunc(df, aG, aS, Rd, Vcmax, j, TPU, gm)
    ap = ApFunc(df, aG, aS, Rd, Vcmax, j, TPU, gm)
    a = np.minimum(ac,np.minimum(aj,ap))
    
    return a
    
def print_param(func, poptA):
    """Prints labeled fitted values for assimilation function.
    
    Keyword arguments:
    func -- function used for fitting.
    poptA -- list of fitted values from func.
    """
    param = {"aG", "aS", "Rd", "Vcmax", "j", "TPU", "gm"}
    # get temperature adjusted values
    adjust_val = [poptA[0], 
                  poptA[1], 
                  poptA[2]/ Rd_func(t_leaf, species),
                  poptA[3] / Vcmax_func(t_leaf, species),
                  poptA[4] / J_func(t_leaf, species),
                  poptA[5] / TPU_func(t_leaf, species),
                  poptA[6] / gm_func(t_leaf, species)]
    # get argument names for printing
    print_args = [arg for arg in func.__code__.co_varnames if str(arg) in param]
    result_df = pd.DataFrame({"Variable": print_args, "T-Leaf": poptA, "25C": adjust_val})
    if len(print_args) == len(poptA):
        my_result = zip(print_args, poptA, adjust_val)
    else:
        raise ValueError("There is an unequal number of results and arguments!")
        exit(1)
    for i in my_result:
        print("{0}: {1} @t-leaf, {2} @25C \n".format(i[0], 
                                                     round(float(i[1]), 3),
                                                     round(float(i[2]), 3)))
    return result_df