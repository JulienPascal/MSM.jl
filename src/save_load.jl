"""
  read_priors(pathToCsv::String)

Function to load priors stored in a csv file. Returns a dictionary. The csv
file should have the following columns "name", "value", "upper_bound", "lower_bound"
"""
function read_priors(pathToCsv::String)

    # initialize and empty Dictionary:
    #---------------------------------
    dictionary = OrderedDict{String,Array{Float64,1}}()

    # Check the existence of the file
    #----------------------------------
    if isfile(pathToCsv) == false
        error(string(pathToCsv," not found."))
    end

    # If file exists
    #----------------------------------
    #dataFrame = CSV.read(pathToCsv)
    dataFrame = CSV.File(open(read, pathToCsv)) |> DataFrame

    # Check whether the csv file has the appropriate columns
    #-------------------------------------------------------
    list_to_check = ["name", "value", "upper_bound", "lower_bound"]
    col_names = names(dataFrame)

    for col_name in list_to_check
        if in(col_name, col_names) == false
            error("column $(col_name) does not exist.")
        end
    end

    # store the number of rows in the dataframe:
    #------------------------------------------
    number_rows = size(dataFrame, 1)

    # Append the dictionary
    #----------------------
    for i=1:number_rows
        dictionary[dataFrame[i,:name]] = [dataFrame[i,:value],  dataFrame[i,:lower_bound], dataFrame[i,:upper_bound]]
    end


    return dictionary

end


"""
  read_empirical_moments(pathToCsv::String)

Function to load empirical moments stored in a csv file. Returns a dictionary.
The csv file should have the following columns "name", "value", "weight"
"""
function read_empirical_moments(pathToCsv::String)

    # initialize and empty Dictionary:
    #---------------------------------
    dictionary = OrderedDict{String,Array{Float64,1}}()

    # Check the existence of the file
    #----------------------------------
    if isfile(pathToCsv) == false
        error(string(pathToCsv," not found"))
    end

    # If file exists:
    #----------------
    #dataFrame =  CSV.read(pathToCsv)
    dataFrame = CSV.File(open(read, pathToCsv)) |> DataFrame

    #check whether the csv file has the columns "name" and "value"
    #-------------------------------------------------------------
    list_to_check = ["name", "value", "weight"]
    col_names = names(dataFrame)

    for col_name in list_to_check
        if in(col_name, col_names) == false
            error("column $(col_name) does not exist.")
        end
    end

    # store the number of rows in the dataframe:
    #------------------------------------------
    number_rows = size(dataFrame,1)


    # Append the dictionary
    #----------------------
    for i=1:number_rows
        dictionary[dataFrame[i,:name]] = [dataFrame[i,:value],  dataFrame[i,:weight]]
    end


    return dictionary

end
