import shapefile ## conda install pyshp   ## details:  https://pypi.python.org/pypi/pyshp
sf = shapefile.Reader("ne_10m_populated_places/ne_10m_populated_places.dbf")
records = sf.records()
fields = sf.fields

## This snippet will export shapefiles records into a CSV (sep='|')
## field_names = SCALERANK|NATSCALE|LABELRANK|FEATURECLA|NAME|NAMEPAR|NAMEALT|DIFFASCII|NAMEASCII|ADM0CAP|CAPALT|CAPIN|WORLDCITY|MEGACITY|SOV0NAME|SOV_A3|ADM0NAME|ADM0_A3|ADM1NAME|ISO_A2|NOTE|LATITUDE|LONGITUDE|CHANGED|NAMEDIFF|DIFFNOTE|POP_MAX|POP_MIN|POP_OTHER|RANK_MAX|RANK_MIN|GEONAMEID|MEGANAME|LS_NAME|LS_MATCH|CHECKME|MAX_POP10|MAX_POP20|MAX_POP50|MAX_POP300|MAX_POP310|MAX_NATSCA|MIN_AREAKM|MAX_AREAKM|MIN_AREAMI|MAX_AREAMI|MIN_PERKM|MAX_PERKM|MIN_PERMI|MAX_PERMI|MIN_BBXMIN|MAX_BBXMIN|MIN_BBXMAX|MAX_BBXMAX|MIN_BBYMIN|MAX_BBYMIN|MIN_BBYMAX|MAX_BBYMAX|MEAN_BBXC|MEAN_BBYC|COMPARE|GN_ASCII|FEATURE_CL|FEATURE_CO|ADMIN1_COD|GN_POP|ELEVATION|GTOPO30|TIMEZONE|GEONAMESNO|UN_FID|UN_ADM0|UN_LAT|UN_LONG|POP1950|POP1955|POP1960|POP1965|POP1970|POP1975|POP1980|POP1985|POP1990|POP1995|POP2000|POP2005|POP2010|POP2015|POP2020|POP2025|POP2050|CITYALT

field_names = '|'.join(str(innerlist[0]) for innerlist in sf.fields).split("|", 1)[1] # first entry is not represented in record
print field_names
with open('./naturalearth.csv','w') as f:
    # f.write('city,latitude,longitude\n')
    f.write(field_names) ## write header
    for record in records:
        # line = "{},{},{}\n".format(record[4], record[21], record[22])
        line = '{}\n'.format('|'.join(str(item) for item in record))
        f.write(line)
