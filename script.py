################################################################################
########################## Colorblind Checking Script ##########################
################################################################################

##### Import packages #####

# --- General manipulations --- #
from collections import Counter # Counter container
import pandas as pd # For dataset management

# --- Image management --- #
from PIL import Image # Import image package of python 

# --- Color management -- #
from colormath.color_objects import sRGBColor, LabColor #Color encodings
from colormath.color_conversions import convert_color # Color converting
from colormath.color_diff import delta_e_cie2000 # Color comparison


##### Get and decompose image #####

# ---Get an image --- #

def getImageRGB(file):
    '''Goal: Transform an image into a PIL image.
    Input: String - URL link of the image.
    Output: PIL Image - RGB converted.
    '''
    
    try:
        img = Image.open(file) # get image
        imgRGB = img.convert('RGB') # convert to rgb color 
        
        return imgRGB
    
    except NameError:
        print("NameError: Use string symbols ('').")
    except FileNotFoundError:
        print('FileNotFoundError: No such file or directory: \'{}\'.'.format(file))
          
          
# --- Extract colors --- #

def getMostCommonColor(img, top_colors = 0.0005, grey_scale = False, approx_grey = True):
    '''Goal: Get most frequent pixel color of a converted RGB image. 
    
    Set top_colors ≥ 1 to keep this number of colors. Set top_colors ∈ ]0,1[ to define the minimum percentage covered by colors to keep.
    Set grey_scale to 'True' to keep grey colors in analysis.
    Set approx_grey to 'True' (in combination to 'grey_scale = False') to also remove non-perfect grey shade (in a rgb point of view). 
    Otherwise if approx_grey is set to 'False' it will remove only perfect shades of grey (r=b=g)
     
    Input: PIL Image - RGB converted.
    Output: Pandas dataframe - Most common colors.
    '''
    
    print('Info : Image size (in px) :', img.size,'.\n\n')
    
    #Get all pixels
    list_color = [img.getpixel((i,j)) for i in range(img.size[0]) for j in range(img.size[1])]
    
    #Count each color
    count = Counter(list_color)
    df = pd.DataFrame.from_dict(count, orient='index').reset_index()
    df.columns = ['color','count']
    df['count_percent'] = df['count'] / (img.size[0] * img.size[1]) * 100
    df_out = df.sort_values(by='count', ascending=False)
    
    #Keep main colors
    try:        
        # Remove shade of grey and approximative shade of grey
        if grey_scale == False & approx_grey == False:
            df_out = df_out[df_out.apply(lambda x: len(set(list(x['color']))) > 1, axis=1)] #Remove grey colors from df
            #Keep top colors
            if top_colors < 1 and top_colors > 0:
                df_out_top = df_out[df.count_percent >= top_colors * 100]
            elif top_colors >= 1:
                df_out_top = df_out.head(top_colors)
                
        if grey_scale == False & approx_grey == True:
            df_out = df_out[df_out.apply(lambda x: np.var(list(x['color'])) > 15, axis=1)] #Remove grey colors from df
            #Keep top colors
            if top_colors < 1 and top_colors > 0:
                df_out_top = df_out[df.count_percent >= top_colors * 100]
            elif top_colors >= 1:
                df_out_top = df_out.head(top_colors)
                
        if grey_scale == True:
            #Keep top colors
            if top_colors < 1 and top_colors > 0:
                df_out_top = df_out[df.count_percent >= top_colors * 100]
            elif top_colors >= 1:
                df_out_top = df_out.head(top_colors)
                
        return df_out_top
    
    except ValueError:
        print("ValueError: Use only numerical and positive values for top_colors.")
    except UnboundLocalError:
        print("ValueError: Use only numerical and positive values for top_colors.")
        
        
 ##### Color construction #####
 
 # --- Difference calculation --- #
 
def colorDifferences(color1_RGB, color2_RGB):
    '''Goal: Compute the color difference between 2 RGB colors
    Input: Tuple - 2 RGB colors
    Output: Value - distance of colors
    '''
    try:
        # Get colors
        color1_rgb = sRGBColor(color1_RGB[0],color1_RGB[1],color1_RGB[2])
        color2_rgb = sRGBColor(color2_RGB[0],color2_RGB[1],color2_RGB[2])

        # Convert from RGB to Lab Color Space
        color1_lab = convert_color(color1_rgb, LabColor)
        color2_lab = convert_color(color2_rgb, LabColor)

        # Compute the color difference
        delta_e = delta_e_cie2000(color1_lab, color2_lab)
    
        return delta_e
    
    except IndexError:
        print("IndexError: Tuples need 3 elements as it represent a RGB encoding")
    except TypeError:
        print("TypeError: Tuples needed as input")
    except ValueError:
        print("ValueError: Use only numerical values in tuples")


 # --- Transform to Colorblind Colors --- #

 def getNewColor(RGBcolor, transform_type=0):
    '''Goal: Transform a RGB tuple color into its equivalent for blindcolor people.
    Set transform_type to '1' for protanopia's vision transformation, '2' for deuteranopia's vision, '3' for tritanopia's vision, '0' for no transform.
    
    Input: Tuple - RGB colors
    Output: Tuple - Colorblind transformed RGB colors
    '''
    
    # Get matrix conversion between rgb to lms encoding, and transforms for the 3 types of colorblind
    rgb2lms = np.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[0.0299566,0.184309,1.46709]])
    prop = np.array([[0,2.02344,-2.52581],[0,1,0],[0,0,1]])
    deut = np.array([[1,0,0],[0.494207,0,1.24827],[0,0,1]])
    trit = np.array([[1,0,0],[0,1,0],[-0.395913,0.801109,0]])
    
    try:
        if min(RGBcolor) < 0:
            raise ValueError()

        #Get RGB encoding into lms encoding
        rgb = np.array([RGBcolor[0]/255.0, RGBcolor[1]/255.0,RGBcolor[2]/255.0])
        LMScolor= np.dot(rgb2lms,rgb)
    
        #Transform into colorblind colors
        if transfrom_type not in (1,2,3,0):
            raise TypeError()
        if transfrom_type == 1:
            LMScolor_cb = np.dot(prop,LMScolor)
        elif transfrom_type == 2:
            LMScolor_cb = np.dot(deut,LMScolor)
        elif transfrom_type == 3:
            LMScolor_cb = np.dot(trit,LMScolor)
        else:
            LMScolor_cb = LMScolor
        
        #Get RGB encoding back
        new_rgb = np.dot(np.linalg.inv(rgb2lms),LMScolor_cb) 
        new_rgb = np.round(new_rgb * 255,1)
        return tuple(new_rgb)

    except TypeError:
        print("TypeError: Use only numerical and positive values in 3-size tuple. For transform_type, use only 1, 2 or3 for colorblind types and 0 for regular vision")
    except ValueError:
        print("TypeError: Use only numerical and positive values in 3-size tuple.")
        ValueError       
        
        
 # --- Create the Colorblind dataframe --- #         
        
def computeColorblind(df_colors, colors_column, transform_type):
    '''Goal : Use the getNewColor() function on the color dataframe to reconstruct new dataframe color for colorblind.
       It return the same dataframe with new colors
       
       Set transform_type to '1' for protanopia's vision transformation, '2' for deuteranopia's vision, '3' for tritanopia's vision, '0' for no transform.
       
       Input: Pandas dataframe - Initial colors
       Output: Pandas dataframe - Transformed colors'''
    
    df = df_colors.copy(deep=True)
    
    df[colors_column] = [getNewColor(x,transform_type) for x in list(df_colors[colors_column])]
    
    return df

##### Analysis functions #####

# --- Compute average color comprehensibility --- #        
        
def colorComprehensibility(df_colors, colors_column, analysis_type):
    '''Goal: Compute the distance between colors
       
       Set analysis_type to 'average' to compute the average distance between colors.
        To 'averagePond' to compute the average distance between colors including with size ponderation
        To 'minDiff' to find the maximum difference between two colors. 
        
    Input: Pandas dataframe - RGB colors
    Output: value - agregated color comprehensibility'''
    
    try:
        #prevent crash if the analysis_type is wrong
        ['average','averagePond','minDiff'].index(analysis_type)
        
        if analysis_type == 'average':
            out = np.mean([colorDifferences(x,y) for i,x in enumerate(df_colors[colors_column].tolist()) for j,y in enumerate(df_colors[colors_column].tolist()) if j>i])
        if analysis_type == 'averagePond':
            out = np.mean([colorDifferences(x,y) * (df_colors.iloc[i,2]/100 + df_colors.iloc[j,2]/100) for i,x in enumerate(df_colors[colors_column].tolist()) for j,y in enumerate(df_colors[colors_column].tolist()) if j>i])
        if analysis_type == 'minDiff':
            out = np.max([colorDifferences(x,y) for i,x in enumerate(df_colors[colors_column].tolist()) for j,y in enumerate(df_colors[colors_column].tolist()) if j>i])
    
        return out

    except ValueError:
        print("ValueError: Use one of the following possibility as a analysis_type : 'average', 'averagePond','minDiff'.")
    except TypeError:
        print("TypeError: df_colors must be a pandas dataframe, generated from the getMostCommonColor() function.")
    except KeyError:
        print("KeyError: invalid column name")
        

# --- Compute Global deviation --- #
 
def colorDeviation(df_colors, colors_column, analysis_type = 'all'):
    '''Goal: Compute de deviation between normal vision and a colorblind vision according to a criteria
    
    Set analysis_type to 'average' to compute the average distance between colors.
        To 'averagePond' to compute the average distance between colors including with size ponderation
        To 'minDiff' to find the maximum difference between two colors.
        To 'all' (default) to perform the 3 previous analysis
        
    Input: Pandas dataframe - Most common colors
    Output: list of color comprehension computation
    '''

    try: 
        #Create color datasets
        pro = computeColorblind(df_colors, colors_column, 1)
        deu = computeColorblind(df_colors, colors_column, 2)
        tri = computeColorblind(df_colors, colors_column, 3)
        
        #prevent crash if the analysis_type is wrong
        ['average','averagePond','minDiff','all'].index(analysis_type)

        
        if analysis_type == 'all':
            avg = colorComprehensibility(df_colors, colors_column, analysis_type = 'average')
            avgpond = colorComprehensibility(df_colors, colors_column, analysis_type = 'averagePond')
            mind = colorComprehensibility(df_colors, colors_column, analysis_type = 'minDiff')
        
            #Pro color
            avgP = colorComprehensibility(pro, colors_column, analysis_type = 'average')
            avgpondP = colorComprehensibility(pro, colors_column, analysis_type = 'averagePond')
            mindP = colorComprehensibility(pro, colors_column, analysis_type = 'minDiff')
            
            #Deu color
            avgD = colorComprehensibility(deu, colors_column, analysis_type = 'average')
            avgpondD = colorComprehensibility(deu, colors_column, analysis_type = 'averagePond')
            mindD = colorComprehensibility(deu, colors_column, analysis_type = 'minDiff')
            
            #Tri color
            avgT = colorComprehensibility(tri, colors_column, analysis_type = 'average')
            avgpondT = colorComprehensibility(tri, colors_column, analysis_type = 'averagePond')
            mindT = colorComprehensibility(tri, colors_column, analysis_type = 'minDiff')
            
            return [avg, avgP, avgD, avgT, avgpond, avgpondP, avgpondD, avgpondT, mind, mindP, mindD, mindT]
        
        if analysis_type == 'average':
            avg = colorComprehensibility(df_colors, colors_column, analysis_type = 'average')
            avgP = colorComprehensibility(pro, colors_column, analysis_type = 'average')
            avgD = colorComprehensibility(deu, colors_column, analysis_type = 'average')
            avgT = colorComprehensibility(tri, colors_column, analysis_type = 'average')
            
            return [avg, avgP, avgD, avgT]
        
            
        if analysis_type == 'averagePond':
            avgpond = colorComprehensibility(df_colors, colors_column, analysis_type = 'averagePond')
            avgpondP = colorComprehensibility(pro, colors_column, analysis_type = 'averagePond')
            avgpondD = colorComprehensibility(deu, colors_column, analysis_type = 'averagePond')
            avgpondT = colorComprehensibility(tri, colors_column, analysis_type = 'averagePond')
            
            return [avgpond, avgpondP, avgpondD, avgpondT]
        
        if analysis_type == 'minDiff':
            mind = colorComprehensibility(df_colors, colors_column, analysis_type = 'minDiff')
            mindP = colorComprehensibility(pro, colors_column, analysis_type = 'minDiff')
            mindD = colorComprehensibility(deu, colors_column, analysis_type = 'minDiff')
            mindT = colorComprehensibility(tri, colors_column, analysis_type = 'minDiff')
        
            return [mind, mindP, mindD, mindT]
        
        if min(RGBcolor) < 0:
            raise ValueError()
    
    except TypeError:
        print("TypeError: df_colors must be a pandas dataframe, generated from the getMostCommonColor() function.")
        
    except ValueError:
        print("ValueError: Use one of the following possibility as a analysis_type :'all', 'average', 'averagePond','minDiff'.")
           
        
# --- Ratio sub-function --- #

def factorFromVect(vector):
    '''Goal: From a vector of length 4, divided the first element by the others
    Input: list of length 4
    Output: list of length 3 - ratio of comprehensibility'''
    try:
        if len(vector) != 4:
            raise IndexError()
        return [vector[0]/vector[1], vector[0]/vector[2], vector[0]/vector[3]]

    except IndexError: 
        print("List must be of length 4")
    except ZeroDivisionError: 
        print("Issues in the color perception value : zero raised")
 

# --- Raise warning for colorblind issues --- #

def colorAnalysis(df_colors, colors_column, analysis_type = 'all', threshold = 5):
    '''Create all possibilities and analyse colors differences
    
    Set analysis_type to 'average' to compute the average distance between colors.
        To 'averagePond' to compute the average distance between colors including with size ponderation
        To 'minDiff' to find the maximum difference between two colors.
        To 'all' (default) to perform the 3 previous analysis
    Set threshold to the minimum ratio you wish not to surpass. 
        For example : 5 to raise a warning if it's 5 times more difficult for a colorblind person to distinguish a color.
    
    Input: Pandas dataframe - Most common colors
    Output: Print of result'''
    isOk = True
    
    try:
        #prevent crash if the analysis_type is wrong
        ['average','averagePond','minDiff','all'].index(analysis_type)
        if int(threshold) < 1:
            raise ValueError()
        
        list_perception = colorDeviation(df_colors, colors_column, analysis_type)
    
        if analysis_type == 'all':
            out = []
            for x in [0,1,2]:
                sub_list_perception = list_perception[x*4:(x+1)*4]
                res = factorFromVect(sub_list_perception)
                measure = ['average distance comprehensibility between colors. It\'s ', 'average distance with ponderation on the area covered comprehensibility between colors. It\'s ','minimum difference comprehensibility between colors. It\'s '][x]
                vision_issues = list([['Issues for protanopia according to ', 'Issues for deuteranopia according to ','Issues for tritanopia according to'][index] for index in [x[0] for x in list(enumerate(res)) if x[1]>threshold]])
                values_issues = list([value for value in res if value > threshold])
                out.append([str(vision_issues[x] + measure + str(round(values_issues[x],1)) + ' times more difficult for them to make the difference.') for x in list(range(len(vision_issues)))])
                
            if len(out) >= 1:
                isOk = False
                outList = [item for sublist in out for item in sublist]
                print('-------\n')
                print('\n'.join(outList))
        
        else:
            if analysis_type == 'average':
                measure = 'average distance comprehensibility between colors. It\'s '
            if analysis_type == 'averagePond':
                measure = 'average distance with ponderation on the area covered comprehensibility between colors. It\'s '
            if analysis_type == 'minDiff':
                measure = 'minimum difference comprehensibility between colors. It\'s '

            res = factorFromVect(list_perception)

            vision_issues = list([['Issues for protanopia according to ', 'Issues for deuteranopia according to ','Issues for tritanopia according to'][index] for index in [x[0] for x in list(enumerate(res)) if x[1]>threshold]])
            values_issues = list([value for value in res if value > threshold])
            
            if len(vision_issues) >= 1:
                isOk = False
                print('-------\n')
                print('\n'.join([str(vision_issues[x] + measure + str(round(values_issues[x],1)) + ' times more difficult for them to make the difference.') for x in list(range(len(vision_issues)))]))
        if isOk == True:
            print('-------\n')
            print('There is no warning due to colorblind about colors used in the image')
            
    except ValueError:
        print("ValueError: Use one of the following possibility as a analysis_type :'all', 'average', 'averagePond','minDiff'. The threshold must be numeric and positive")

        
##### Run an example #####

df_colors = getMostCommonColor(getImageRGB("/Users/eliotm/Documents/duck.png"),top_colors = 10, grey_scale = False, approx_grey = True)
colorAnalysis(df_colors, 'color', analysis_type = 'all', threshold = 2)   

