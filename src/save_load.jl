# Function to read priors from a csv file. returns a Dict
#
# input :
# -------
# * path to csv file
#
# the csv file should contain at least two columns with:
# * "name" (name of the parameter)
# * "value" (value of the parameter)
# * "upper_bound" (upper bound for the support of parameter "name")
# * "lower_bound" (upper bound for the support of parameter "name")
#
# output :
# -------
# * a dictionary with "name" => [value, lower_bound, upper_bound]
#
# Remark:
#--------
# *
#----------------------------------------------------------------------------
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
   dataFrame = CSV.read(pathToCsv)

   # Check whether the csv file has the appropriate columns
   #-------------------------------------------------------
   ListToCheck = [:name, :value, :upper_bound, :lower_bound]

   for colName in ListToCheck
     try
       dataFrame[colName]
     catch
       error("column $(colName) does not exist.")
     end
   end

   # store the number of rows in the dataframe:
   #------------------------------------------
   number_rows = size(dataFrame, 1)
   info(string(number_rows, " prior value(s) found"))

   # Append the dictionary
   #----------------------
   for i=1:number_rows
      dictionary[dataFrame[i,:name]] = [dataFrame[i,:value],  dataFrame[i,:lower_bound], dataFrame[i,:upper_bound]]
   end


   return dictionary

 end

#----------------------------------------------------------------------------
# Function to read moments and weights from a csv file. returns a DataFrame
#
# input :
# -------
# * path to csv file
#
# the csv file should contain at least two columns with:
# * "name" (name of the parameter)
# * "value" (value of the parameter)
# * "weight" (weight associated to moment "name")
#
# output :
# -------
# a dictionary with "name" => [value, weights]
#
# Remark:
#--------
#
#----------------------------------------------------------------------------
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
  dataFrame =  CSV.read(pathToCsv)

  #check whether the csv file has the columns "name" and "value"
  #-------------------------------------------------------------
  ListToCheck = [:name, :value, :weight]

  for colName in ListToCheck
   try
     dataFrame[colName]
   catch
     error("column $(colName) does not exist.")
   end
  end

  # store the number of rows in the dataframe:
  #------------------------------------------
  number_rows = size(dataFrame,1)
  info(string(number_rows, " moment(s) found."))


  # Append the dictionary
  #----------------------
  for i=1:number_rows
     dictionary[dataFrame[i,:name]] = [dataFrame[i,:value],  dataFrame[i,:weight]]
  end


  return dictionary

end

"""
  saveSMMOptim(sMMProblem::SMMProblem; verbose::Bool = false, saveName::String = "")

Function to save a SMMProblem to disk.
"""
function saveSMMOptim(sMMProblem::SMMProblem; verbose::Bool = false, saveName::String = "")

    bestValue = best_candidate(sMMProblem.bbResults)
    bestFitness = best_fitness(sMMProblem.bbResults)

    if verbose == true
        info("Best Value: ", bestValue)
        info("Best Fitness: ", bestFitness)
        info("Saving optimisation to disk")
    end

    # If no name is provided, generate a random name
    #-----------------------------------------------
    if isempty(saveName) == true
        saveName = string(rand(1:Int(1e8)))
    end


    tempfilename = saveName * ".jld2"
    JLD2.@save tempfilename sMMProblem


    if verbose == true
        info("Done.")
    end

end

"""
  loadSMMOptim(saveName::String; verbose::Bool = false)

Function to save a SMMProblem to disk.
"""
function loadSMMOptim(saveName::String; verbose::Bool = false)

    if verbose == true
        info("Loading optimisation from disk")
    end

    #-------------------------------
    # Load BBsetup and BBoptimize
    #--------------------------------
    if isempty(saveName) == true
        error("Please enter a non-empty saveName.")
    end

    tempfilename = saveName * ".jld2"
    JLD2.@load tempfilename sMMProblem

    if verbose == true
        info("Done.")
    end

    return sMMProblem

end
