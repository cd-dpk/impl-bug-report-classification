def splitCamelCase( str):
 	   return str.replace(("%s|%s|%s","(?<=[A-Z])(?=[A-Z][a-z])","(?<=[^A-Z])(?=[A-Z])",
           "(?<=[A-Za-z])(?=[^A-Za-z])")," ")
tests = [
        "camel",
        "camelCase",
        "CamelCase",
        "CAMELCASE",
        "camelcase",
        "Camelcase",
        "Case"
        ]

for test in tests:
    print (test, splitCamelCase(test))