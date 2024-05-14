Search.setIndex({"docnames": ["index", "reference/api/operation.Operation._mulvar_helper", "reference/api/operation.Operation.add_log", "reference/api/operation.Operation.diff", "reference/api/operation.Operation.log", "reference/api/operation.Operation.rolling_max", "reference/api/operation.Operation.rolling_min", "reference/api/operation.Operation.scale", "reference/api/operation.Operation.shift", "reference/api/operation.Operation.signed_power", "reference/api/operation.Operation.sub_log", "reference/index", "reference/operation"], "filenames": ["index.rst", "reference/api/operation.Operation._mulvar_helper.rst", "reference/api/operation.Operation.add_log.rst", "reference/api/operation.Operation.diff.rst", "reference/api/operation.Operation.log.rst", "reference/api/operation.Operation.rolling_max.rst", "reference/api/operation.Operation.rolling_min.rst", "reference/api/operation.Operation.scale.rst", "reference/api/operation.Operation.shift.rst", "reference/api/operation.Operation.signed_power.rst", "reference/api/operation.Operation.sub_log.rst", "reference/index.rst", "reference/operation.rst"], "titles": ["Welcome to Alpha documentation", "operation.Operation._mulvar_helper", "operation.Operation.add_log", "operation.Operation.diff", "operation.Operation.log", "operation.Operation.rolling_max", "operation.Operation.rolling_min", "operation.Operation.scale", "operation.Operation.shift", "operation.Operation.signed_power", "operation.Operation.sub_log", "API Reference", "Operation"], "terms": {"date": [0, 3, 5, 6], "mai": 0, "14": 0, "2024": 0, "download": 0, "zip": 0, "html": 0, "us": [0, 1, 3, 8], "link": 0, "panda": [0, 8], "numpi": [0, 1, 3, 4, 5, 6, 7, 8, 9], "xarrai": [0, 5, 6], "The": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "refer": 0, "guid": 0, "contain": [0, 1], "detail": [0, 12], "descript": 0, "api": 0, "describ": 0, "how": 0, "method": [0, 2, 5, 6, 10, 11], "work": 0, "which": [0, 1, 8, 9], "paramet": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "can": [0, 8], "It": [0, 5, 6], "assum": 0, "you": 0, "have": 0, "an": [0, 4, 5, 6, 8, 11], "understand": 0, "kei": 0, "concept": 0, "commonli": 0, "python": 0, "base": [0, 4], "data": [0, 4, 8, 12], "scientif": 0, "librari": [0, 5, 6, 12], "classmethod": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "xmatrix": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "ndarrai": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "ani": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "dtype": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "ymatrix": [1, 2, 10], "tupl": 1, "helper": [1, 11], "function": [1, 4, 9, 11], "multi": 1, "variabl": 1, "handl": [1, 4, 5, 6], "nan": [1, 3, 4, 5, 6, 7, 8], "valu": [1, 3, 4, 7, 8, 9], "thi": [1, 2, 5, 6, 8, 9, 10, 11], "class": [1, 11], "mathod": 1, "indentifi": 1, "postion": 1, "where": 1, "either": 1, "return": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "copi": [1, 8], "input": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "place": 1, "posit": [1, 3, 8], "both": [1, 2, 3, 10], "nd": 1, "arrai": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "array_lik": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "matrix": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "must": 1, "match": 1, "shape": [1, 2, 10], "y": [1, 2, 10], "A": 1, "two": [1, 2, 10], "cp_x": 1, "cp_y": 1, "ar": [1, 8, 11], "origin": 1, "insert": 1, "type": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "exampl": [1, 3, 4, 5, 6, 7, 8, 9, 12], "import": [1, 3, 4, 5, 6, 7, 8, 9], "np": [1, 3, 4, 5, 6, 7, 8, 9], "1": [1, 3, 4, 5, 6, 7, 8, 9], "0": [1, 3, 4, 7, 8, 9], "2": [1, 3, 5, 6, 7, 8, 9], "4": [1, 3, 5, 6, 7, 8, 9], "5": [1, 3, 5, 6, 7, 8, 9], "3": [1, 3, 5, 6, 7, 8, 9], "8": [1, 3, 5, 6, 8], "creat": 1, "avoid": 1, "modifi": 1, "identifi": 1, "copyto": 1, "argument": 1, "calcul": [2, 3, 4, 5, 6, 7, 10], "element": [2, 3, 4, 7, 9, 10, 11], "wise": [2, 4, 7, 9, 10, 11], "natur": [2, 4, 10], "logarithm": [2, 4, 10], "sum": [2, 7], "like": [2, 9, 10], "object": [2, 8, 9, 10, 11], "take": [2, 9, 10], "same": [2, 7, 8, 10], "comput": [2, 5, 6, 10], "each": [2, 4, 5, 6, 7, 9, 10], "period": [3, 8], "int": [3, 5, 6, 7, 8, 9], "fill_valu": [3, 8], "none": [3, 8], "n": 3, "th": 3, "discret": 3, "differ": [3, 10], "along": 3, "dimens": [3, 5, 6, 7, 8], "fill": [3, 8], "miss": [3, 8], "produc": 3, "shift": 3, "ones": 3, "final": 3, "result": [3, 7], "accept": 3, "neg": [3, 8], "integ": 3, "scalar": [3, 8], "option": [3, 5, 6, 8], "newli": [3, 8], "By": [3, 5, 6, 8], "default": [3, 5, 6, 8], "7": [3, 5, 6, 8], "6": [3, 5, 6, 8, 9], "9": [3, 5, 6, 8, 9], "one": [4, 8], "plu": 4, "inf": 4, "doc": 4, "log1p": 4, "For": 4, "real": 4, "accur": 4, "also": 4, "so": 4, "small": 4, "float": [4, 7, 9], "point": 4, "accuraci": 4, "multivalu": 4, "infinit": 4, "number": [4, 5, 6, 8], "complex": 4, "z": 4, "exp": [4, 9], "alwai": 4, "output": 4, "cannot": 4, "express": 4, "infin": 4, "yield": 4, "set": 4, "invalid": 4, "error": 4, "flag": 4, "see": [4, 12], "seterr": 4, "equival": 4, "1e": 4, "99": 4, "window_s": [5, 6], "minob": [5, 6], "roll": [5, 6, 11], "maximum": [5, 6], "util": [5, 6], "window": [5, 6, 11], "first": [5, 6], "convert": [5, 6], "dataarrai": [5, 6], "ticker": [5, 6], "Then": [5, 6], "across": [5, 6], "requir": [5, 6], "minimum": [5, 6], "observ": [5, 6], "specifi": [5, 6], "size": [5, 6], "10": [5, 6], "11": [5, 6], "12": [5, 6], "rolling_max": 6, "union": [7, 9], "given": 7, "factor": 7, "divis": 7, "absolut": [7, 9], "row": 7, "ignor": 7, "6666667": 7, "3333333": 7, "833333": 7, "index": 8, "desir": 8, "onli": 8, "move": 8, "while": 8, "axi": 8, "label": 8, "remain": 8, "consist": 8, "behavior": 8, "datafram": 8, "from": 8, "beyond": 8, "bound": 8, "appear": 8, "end": 8, "accord": 8, "introduc": 8, "get": 9, "sign": 9, "exponenti": 9, "power": 9, "binari": 9, "pow": 9, "multipli": 9, "rais": 9, "expon": 9, "s": 9, "should": 9, "16": 9, "25": 9, "36": 9, "41421356": 9, "73205081": 9, "23606798": 9, "44948974": 9, "page": 11, "give": 11, "overview": 11, "all": 11, "public": 11, "alphadev": 11, "expos": 11, "alpha": [11, 12], "factorpool": 11, "oper": 11, "namespac": 11, "follow": 11, "submodul": 11, "pair": 11, "form": 12, "cornerston": 12, "serv": 12, "defin": 12, "mathemat": 12, "manipul": 12, "appli": 12, "further": 12, "usag": 12}, "objects": {"operation.Operation": [[1, 0, 1, "", "_mulvar_helper"], [2, 0, 1, "", "add_log"], [3, 0, 1, "", "diff"], [4, 0, 1, "", "log"], [5, 0, 1, "", "rolling_max"], [6, 0, 1, "", "rolling_min"], [7, 0, 1, "", "scale"], [8, 0, 1, "", "shift"], [9, 0, 1, "", "signed_power"], [10, 0, 1, "", "sub_log"]]}, "objtypes": {"0": "py:method"}, "objnames": {"0": ["py", "method", "Python method"]}, "titleterms": {"welcom": 0, "alpha": 0, "document": 0, "fast": 0, "index": 0, "recent": 0, "chang": 0, "oper": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], "_mulvar_help": 1, "add_log": 2, "diff": 3, "log": 4, "rolling_max": 5, "rolling_min": 6, "scale": 7, "shift": 8, "signed_pow": 9, "sub_log": 10, "api": 11, "refer": 11, "helper": 12, "function": 12, "element": 12, "wise": 12, "pair": 12, "roll": 12, "window": 12}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 56}})