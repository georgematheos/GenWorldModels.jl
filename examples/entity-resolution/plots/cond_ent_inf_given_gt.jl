### Conditional entropy of inferred relations given ground truth ###

### GenWorldModels split/merge: ###
times_gwm = [164.928, 1731.714, 172.088, 179.136, 187.246, 194.181, 11.038, 201.203, 208.247, 215.36, 222.595, 229.953, 237.017, 244.333, 251.578, 258.713, 265.778, 17.45, 272.833, 280.738, 289.882, 298.811, 305.997, 313.375, 320.927, 328.002, 335.055, 342.183, 23.98, 349.161, 356.216, 363.095, 370.053, 377.044, 383.883, 390.838, 397.693, 404.521, 411.343, 30.827, 418.413, 425.192, 432.194, 439.13, 445.95, 452.918, 459.839, 466.655, 473.296, 480.119, 37.609, 487.054, 493.874, 500.602, 507.278, 514.568, 521.215, 527.804, 534.597, 541.38, 548.175, 44.7, 554.878, 561.555, 568.163, 575.215, 582.109, 589.042, 595.828, 602.495, 609.392, 616.446, 52.105, 623.328, 630.151, 637.094, 644.511, 651.636, 658.723, 665.74, 672.833, 679.83, 686.65, 0.0, 59.135, 693.463, 700.316, 706.957, 713.505, 720.212, 727.165, 733.972, 740.884, 747.743, 754.568, 66.152, 761.404, 768.317, 775.111, 781.857, 788.813, 795.767, 802.705, 809.532, 816.497, 823.351, 72.971, 830.133, 837.014, 843.86, 850.862, 857.791, 864.559, 871.56, 878.275, 885.176, 892.077, 80.001, 899.051, 906.077, 912.906, 919.858, 926.803, 933.482, 940.184, 946.978, 954.161, 960.848, 86.973, 967.548, 974.211, 981.078, 987.929, 994.897, 1001.847, 1008.969, 1015.854, 1022.742, 1029.65, 93.878, 1036.558, 1043.87, 1050.846, 1057.775, 1064.986, 1072.001, 1078.806, 1085.838, 1092.96, 1099.92, 100.852, 1106.954, 1113.989, 1120.833, 1127.73, 1134.618, 1141.583, 1148.436, 1155.252, 1162.224, 1169.025, 108.184, 1175.931, 1182.809, 1189.906, 1196.863, 1203.968, 1211.212, 1218.123, 1225.111, 1232.249, 1239.377, 115.057, 1246.311, 1253.242, 1260.399, 1267.638, 1274.834, 1283.015, 1290.118, 1297.593, 1304.586, 1311.543, 122.247, 1318.573, 1325.432, 1332.418, 1339.361, 1346.42, 1353.472, 1360.705, 1367.709, 1374.561, 1381.545, 5.192, 129.24, 1388.654, 1395.679, 1402.65, 1409.538, 1416.525, 1423.657, 1430.394, 1437.578, 1444.361, 1451.217, 136.469, 1457.927, 1464.619, 1471.455, 1478.635, 1485.432, 1492.5, 1499.31, 1506.255, 1513.207, 1520.457, 143.424, 1527.615, 1534.513, 1541.546, 1548.536, 1555.437, 1562.457, 1569.214, 1576.13, 1583.1, 1590.106, 150.62, 1596.956, 1603.878, 1610.771, 1617.577, 1624.244, 1631.101, 1637.678, 1644.38, 1650.967, 1657.488, 157.622, 1664.415, 1671.141, 1678.131, 1684.75, 1691.337, 1697.995, 1704.653, 1711.439, 1718.148, 1724.883]
ents_gwm = Any[1.7012633610499417, 1.3137778584216087, 1.7052968021366013, 1.6973077350720691, 1.6952578077979532, 1.6835410751355016, 2.2982206083252064, 1.6782243150321179, 1.6756351763800514, 1.6717178658645042, 1.6644194939185573, 1.6597103004858436, 1.643963930676693, 1.646448037729051, 1.6374845835758354, 1.6442646115019668, 1.6352721134508472, 2.22138586721493, 1.6229546999339897, 1.6164709455314936, 1.6122593033234585, 1.6060742480539212, 1.6080302028385653, 1.5976276951995545, 1.5858023722486245, 1.5803784412091582, 1.5817246325621368, 1.5758236539487136, 2.129610665775045, 1.5717228933629919, 1.5737848256460012, 1.5639714406409786, 1.5586808367056955, 1.5556928315837444, 1.5494651702622788, 1.5373975405548321, 1.5267883144073524, 1.5256836956221855, 1.5277886357815156, 2.06813748926824, 1.5229322514856412, 1.5182970293605873, 1.5215095436321775, 1.5129141574005858, 1.5292791131294756, 1.5317298088434579, 1.5280249524348208, 1.5270297725510007, 1.5272403246879311, 1.5244856410310244, 2.042611528190589, 1.5206554570498516, 1.515691650246775, 1.4948579523417886, 1.4991150609309272, 1.4958062231235385, 1.5106132675812431, 1.5049800177037322, 1.5049398002714274, 1.4988747216620015, 1.5019020381787687, 2.016911314044187, 1.5083262649359561, 1.5063545444223527, 1.4963688083185607, 1.497799375623066, 1.5036954501125404, 1.493135149461275, 1.4865855951315516, 1.4866417012350452, 1.4828574304234514, 1.4838280245267124, 1.9669264129975692, 1.4787678941878368, 1.4829669947035011, 1.4772100981545289, 1.4802353584159844, 1.481691955696173, 1.472365894280085, 1.464812948824942, 1.475069261533283, 1.4786467739752402, 1.4833579092391016, 2.537233884621176, 1.9403273425031462, 1.47350141907197, 1.4748184919640477, 1.4721731327843244, 1.4673075514870266, 1.4650292068718143, 1.4709591895219742, 1.4630954500299773, 1.454706435045533, 1.4524600567598587, 1.4558761941322196, 1.906056805074974, 1.4510266159302743, 1.4553891188642183, 1.4483903275595573, 1.4427092125086518, 1.4438214579077016, 1.4421007398004926, 1.4397017319338268, 1.4365411164057906, 1.438363406997597, 1.4379420896559558, 1.8813382730486992, 1.4397347797133908, 1.435944698740359, 1.429480030528547, 1.4288406518136703, 1.42749936458464, 1.4348688599696093, 1.4393639612089186, 1.438373685449047, 1.4375106392316916, 1.4240446290740694, 1.8459160933091254, 1.4295898065185488, 1.4308356305929015, 1.4318821268804305, 1.433860890854357, 1.4260516939423569, 1.4231713112693078, 1.4214699299938593, 1.4124680349996168, 1.4109020843356168, 1.4088696051656908, 1.8249982522897714, 1.405815340867498, 1.4047847222966188, 1.3979224719600611, 1.4030427521452584, 1.4027900806060427, 1.3998630458503662, 1.3987875455260999, 1.393067626075356, 1.403273225741666, 1.3979173296560046, 1.801753637981301, 1.396461813825843, 1.3981186847915115, 1.4036526887015908, 1.4001845007368166, 1.3873568290630396, 1.3924126665244407, 1.3888079375915634, 1.3857018600673519, 1.3846844931952624, 1.3843868145239573, 1.779455352623433, 1.37805259107641, 1.3788706301616758, 1.3864706100975916, 1.3883539109168404, 1.3869676165557205, 1.380244305979557, 1.377916175487009, 1.3876836363679137, 1.384956054777764, 1.3783643810457553, 1.7625773018235256, 1.3786779085270822, 1.3765457588425396, 1.3800399885512504, 1.3836326437074389, 1.3749096806720087, 1.3740035691537114, 1.37111675183034, 1.367905886599782, 1.3699407566772024, 1.362430069129129, 1.7520552022104203, 1.365669925270342, 1.354159451867346, 1.3501569167874035, 1.3477289158385395, 1.3459645079415046, 1.3479186751803323, 1.3420243882366747, 1.3442447247784128, 1.3447142577261777, 1.3424959375785839, 1.7365372896035334, 1.3446871556594213, 1.346631828669338, 1.3397716056814635, 1.3381143962693116, 1.3414395756185364, 1.3392549991736298, 1.3354375175485256, 1.3382101062707654, 1.3358755411278953, 1.3304565398688197, 2.398793718913466, 1.7522859131732231, 1.3316753872551188, 1.3386807118537183, 1.3405297409534704, 1.3377571522312306, 1.3306946070819863, 1.336700854661512, 1.3356625838203078, 1.3361858319640723, 1.3379960886403572, 1.3367438299156826, 1.7406160896128418, 1.3318942517137373, 1.337421524092884, 1.342626903463773, 1.3433961314769876, 1.3485433883502387, 1.3461686057750637, 1.3451530154942521, 1.3432545719607716, 1.3397672691485205, 1.3407201712542287, 1.726453327993962, 1.3287425558109622, 1.3256370895332248, 1.3296125225464108, 1.3263233773501717, 1.3248480544846963, 1.3233543376163739, 1.3250359448536324, 1.324868721901131, 1.3280213261292666, 1.3257961998877434, 1.713158639658618, 1.3266592461050988, 1.3245643515600602, 1.3255948447522365, 1.328636108109045, 1.320571013481541, 1.3230570225038794, 1.3275505755145613, 1.3274122579551806, 1.3206872694313565, 1.3234526029833207, 1.7129059681194023, 1.3220615189224156, 1.315666296201965, 1.3099919473838213, 1.3098888177540933, 1.3115043035061815, 1.3126960488257235, 1.3130490994118185, 1.3134029587496139, 1.3135171873507456, 1.31465081017265]
### Vanilla Gen split/merge: ###
times_van_sm = [993.261, 999.988, 1004.566, 73.896, 1011.351, 1017.696, 1022.333, 1027.8, 1033.707, 1039.389, 1046.828, 1053.435, 1060.113, 1069.687, 75.55, 1076.664, 1081.547, 1087.712, 1095.448, 1101.188, 1108.953, 1116.03, 1120.509, 1125.457, 1131.651, 82.135, 1137.435, 1144.882, 1150.927, 1157.495, 1165.323, 1172.803, 1179.763, 1189.312, 1195.033, 1201.208, 2.645, 88.993, 1208.322, 1210.652, 1219.905, 1227.339, 1234.9, 1242.82, 1248.94, 1256.567, 1264.069, 1269.766, 94.626, 1278.024, 1287.273, 1296.048, 1303.525, 1310.913, 1318.149, 1322.28, 1327.53, 1336.048, 1341.285, 100.45, 1348.254, 1353.494, 1360.398, 1365.23, 1370.518, 1377.057, 1384.924, 1390.92, 1395.823, 1402.009, 107.123, 1409.851, 1416.567, 1425.409, 1431.353, 1439.381, 1445.762, 1450.415, 1456.772, 1463.176, 1471.206, 115.572, 1477.66, 1481.584, 1485.483, 1494.328, 1499.434, 1505.327, 1510.827, 1518.449, 1523.238, 1529.597, 118.294, 1537.511, 122.775, 128.937, 135.888, 142.247, 5.698, 147.93, 152.893, 158.152, 162.948, 168.414, 173.834, 180.614, 185.732, 192.074, 197.971, 8.821, 203.641, 208.474, 214.094, 218.9, 224.728, 229.545, 236.291, 241.878, 248.618, 254.14, 11.162, 260.085, 266.764, 274.583, 280.169, 285.382, 290.896, 296.873, 302.456, 308.817, 317.466, 16.218, 321.909, 326.74, 333.378, 337.761, 344.945, 351.655, 357.586, 363.092, 370.906, 375.337, 20.387, 378.963, 386.063, 393.045, 400.623, 407.487, 413.616, 417.387, 424.717, 430.911, 437.786, 25.777, 443.65, 449.092, 454.861, 458.636, 465.59, 472.177, 478.726, 486.887, 493.656, 502.695, 30.526, 506.174, 511.548, 517.299, 524.943, 529.083, 533.635, 540.06, 546.169, 552.98, 558.189, 0.0, 36.395, 565.109, 570.782, 576.626, 584.317, 589.684, 596.245, 602.784, 611.401, 613.908, 620.52, 42.461, 628.622, 633.602, 637.381, 641.531, 646.908, 654.044, 658.628, 666.779, 673.761, 678.8, 46.695, 684.182, 689.172, 694.179, 700.318, 705.706, 713.467, 720.119, 727.116, 732.966, 736.846, 51.883, 744.695, 753.453, 758.829, 765.904, 770.635, 778.183, 783.361, 788.477, 797.31, 802.466, 56.64, 810.596, 817.025, 822.215, 828.203, 833.992, 841.625, 850.176, 856.662, 861.979, 869.157, 62.664, 876.237, 883.267, 890.962, 895.91, 902.159, 905.778, 912.071, 919.279, 925.532, 933.142, 68.351, 940.263, 945.239, 953.734, 961.763, 970.612, 976.509, 984.971]
ents_van_sm = Any[1.7128225311008236, 1.7155258983540216, 1.7079467967307784, 2.3545756464196628, 1.7050469784660223, 1.7072963190444974, 1.706773070900733, 1.7027976378875471, 1.6967292122993032, 1.696389414225712, 1.6960496161521215, 1.6874631753508336, 1.685824209450498, 1.684437915089378, 2.3275293553444456, 1.6840981170157876, 1.6813255282935475, 1.6799392339324277, 1.6799392339324277, 1.6793177602828844, 1.6800825809944606, 1.6796445574150904, 1.6782582630539709, 1.6757562509362798, 1.667653901169272, 2.309663876653304, 1.6707662879651024, 1.6711864064791393, 1.6686843943614482, 1.6686843943614478, 1.6686843943614478, 1.672221803795264, 1.6747238159129552, 1.6747238159129552, 1.6780196527789593, 1.6787263509928976, 2.6566057273467982, 2.2925050961761007, 1.677340056631778, 1.6756139641970673, 1.672599802907016, 1.672599802907016, 1.6695745426455602, 1.669827214184776, 1.667577873606301, 1.6653285330278254, 1.6647360634150192, 1.663971242703443, 2.274483269481542, 1.6642239142426585, 1.6628376198815389, 1.6633608680253034, 1.6614513255204189, 1.6595417830155343, 1.6595417830155343, 1.6580572631486354, 1.6580572631486354, 1.6580572631486354, 1.6599668056535197, 2.265485907167641, 1.6599668056535197, 1.6569415453920642, 1.6569415453920642, 1.6569415453920642, 1.6560784991747088, 1.6560784991747088, 1.6558258276354934, 1.6596449126452624, 1.6581603927783635, 1.656250850273479, 2.251622963556442, 1.656250850273479, 1.6540015096950038, 1.6540015096950038, 1.6517521691165282, 1.6517521691165282, 1.6540015096950034, 1.6537488381557879, 1.6537488381557879, 1.6537488381557879, 1.651246826038097, 2.2416662314893867, 1.6523625437946678, 1.6498605316769768, 1.6498605316769768, 1.6515174026426458, 1.6529036970037658, 1.6515174026426458, 1.6487269088550731, 1.6462359957087858, 1.6448497013476657, 1.6448497013476657, 2.2250486042212816, 1.6468865878192303, 2.2146378454795044, 2.198681909293247, 2.195431079559333, 2.181539017487233, 2.63615111000359, 2.1763336381163443, 2.1650598331572115, 2.1637427602651336, 2.149540018580344, 2.135424403429929, 2.125467671362874, 2.1194684672436717, 2.110714579467563, 2.094166173668499, 2.0878271714757064, 2.597831013969241, 2.086440877114587, 2.0701451428547384, 2.0673725541324988, 2.0626904229053746, 2.057485043534486, 2.0434546529484665, 2.0321808479893337, 2.026353994968901, 2.0219424403463253, 2.0135104455784747, 2.565440900585053, 2.008897535820392, 2.001373594401986, 2.000510548184631, 1.9922620034869534, 1.9906230375866178, 1.9851247692442089, 1.9746447890333891, 1.9726370210227255, 1.9685923665404979, 1.9659069043526332, 2.5400514804755066, 1.9633177657005674, 1.957548920753609, 1.9566858745362539, 1.9531484651024376, 1.9509683459930043, 1.9448999204047601, 1.9380826772002926, 1.9389457234176481, 1.932917515261709, 1.9265092915998738, 2.517687320627414, 1.924076500951225, 1.9199176178678654, 1.9182786519675294, 1.9194814962584752, 1.9215744888335335, 1.9206422211471363, 1.9154078377395103, 1.9112489546561509, 1.9146832344602387, 1.914499784390065, 2.5065969657384546, 1.9142471128508494, 1.9111347260550189, 1.9119285508033323, 1.8960176217491653, 1.8959484002801235, 1.89343038506708, 1.8967262219330843, 1.8910828189828253, 1.8836280990334615, 1.8777431235153907, 2.4912334991650047, 1.8678904669767198, 1.8687535131940753, 1.8684137151204843, 1.859302124205452, 1.8552124625911344, 1.8510535795077743, 1.8476595171359913, 1.8476595171359913, 1.8533580802910872, 1.8527656106782806, 2.683991816495605, 2.473211672470446, 1.8546751531831651, 1.8526092626748636, 1.8485375061258789, 1.8550149512567558, 1.8549167257509769, 1.8505922976627767, 1.848627594953055, 1.842577074430144, 1.8420538262863793, 1.8354621525543706, 2.4537072278789687, 1.8345991063370155, 1.8307107998582044, 1.8262702411988916, 1.8255233255526488, 1.8210246443956983, 1.815410245482177, 1.8146164207338635, 1.8135007029772925, 1.8107281142550524, 1.8109807857942684, 2.4376820702236692, 1.808731445215793, 1.807345150854673, 1.8048431387369823, 1.8020013285457006, 1.798790716244091, 1.7965413756656154, 1.796018127521851, 1.7881805441979544, 1.7857477535493058, 1.7775684303206702, 2.419816591532528, 1.7735479901753946, 1.769389107092035, 1.7686131874090545, 1.7698271306714044, 1.7640292816877092, 1.76375870508316, 1.764011376622376, 1.7640113766223757, 1.763671578548785, 1.7632175518740627, 2.4056138498477386, 1.7621018341174917, 1.7586225471813135, 1.7571269283430104, 1.7586114482099098, 1.7607044407849677, 1.7607044407849677, 1.7581444205938026, 1.7567581262326826, 1.7523576705815112, 1.748793159080939, 2.3924305023837213, 1.7487931590809385, 1.749754430804073, 1.7473908616244658, 1.7414366646373534, 1.7376175796275843, 1.7365018618710133, 1.734339647826913, 1.7295882951307469, 1.7284546723088428, 1.728541798843218, 2.3779300820276257, 1.7266322563383334, 1.7252459619772138, 1.7260058785648411, 1.724872255742937, 1.722168888489739, 1.719666876372048, 1.7166416161105924]
### Vanilla Gen generic inference: ###
times_van_generic = [872.092, 85.449, 878.287, 884.472, 890.634, 897.023, 903.114, 909.251, 915.408, 921.685, 928.044, 934.083, 91.858, 940.141, 946.357, 952.764, 958.826, 965.015, 971.134, 977.237, 983.366, 989.497, 995.633, 98.506, 1001.897, 1008.083, 1014.08, 1020.16, 1026.15, 1032.356, 1038.447, 1044.599, 1050.64, 1056.905, 104.873, 1063.278, 1069.31, 1075.498, 1081.657, 1087.685, 1093.842, 1099.843, 1105.835, 1111.836, 1117.868, 111.263, 1124.157, 1130.132, 1136.261, 1142.314, 1148.384, 1154.616, 1160.767, 1166.926, 1172.879, 1179.097, 117.611, 1185.033, 1191.09, 1197.078, 1203.157, 1209.225, 1215.607, 1221.684, 1227.667, 1233.966, 1239.997, 6.675, 124.334, 1246.163, 1252.252, 1258.353, 1264.342, 1270.753, 1276.736, 1282.754, 1288.756, 1294.771, 1300.964, 130.661, 1306.855, 1312.856, 1318.925, 1324.947, 1331.223, 1337.085, 1343.179, 1349.289, 1355.382, 1361.746, 137.016, 1367.811, 1373.85, 1379.784, 1385.789, 1391.935, 1397.973, 1404.006, 1410.016, 1415.971, 1422.375, 143.415, 1428.238, 1434.172, 1440.069, 1446.057, 1452.18, 1458.202, 1464.119, 1469.899, 1475.908, 1482.115, 149.783, 1488.157, 1494.157, 1499.984, 1506.041, 1511.949, 1518.132, 1524.144, 1530.224, 1536.272, 1542.614, 156.339, 1548.525, 162.866, 169.395, 175.926, 181.987, 13.412, 188.266, 194.627, 201.061, 207.419, 213.696, 220.201, 226.852, 233.297, 239.688, 245.916, 20.069, 252.501, 258.768, 265.191, 271.562, 277.905, 284.198, 290.529, 296.876, 303.389, 309.672, 26.719, 315.996, 322.396, 329.002, 335.196, 341.464, 347.701, 354.034, 360.425, 366.829, 373.094, 33.368, 379.423, 385.622, 391.752, 397.938, 404.126, 410.437, 416.661, 422.922, 429.227, 435.649, 39.889, 441.686, 447.909, 454.277, 460.583, 466.825, 473.117, 479.367, 485.919, 492.189, 498.333, 46.403, 504.7, 510.859, 517.39, 523.585, 529.791, 536.102, 542.457, 548.629, 554.856, 561.045, 52.984, 567.302, 573.51, 579.729, 586.012, 592.6, 598.798, 604.953, 611.328, 617.438, 623.62, 0.0, 59.385, 629.72, 635.907, 642.177, 648.646, 654.836, 661.05, 667.342, 673.531, 679.826, 686.139, 65.762, 692.443, 698.733, 705.271, 711.48, 717.731, 723.924, 730.112, 736.132, 742.475, 748.735, 72.256, 755.167, 761.194, 767.207, 773.398, 779.681, 785.701, 792.01, 798.139, 804.321, 810.8, 79.049, 816.785, 822.927, 828.893, 835.077, 841.27, 847.415, 853.616, 859.782, 866.061]
ents_van_generic = Any[2.0283554537079453, 2.437315170083322, 2.032416111285526, 2.0257954335167807, 2.026998277807727, 2.0204758255447604, 2.0218621199058804, 2.022483593555424, 2.026032101960644, 2.025479849780142, 2.021068295157567, 2.0241985870187302, 2.4241318226193043, 2.0274944238847348, 2.026108129523615, 2.026038908054573, 2.0262513621614837, 2.026476931633943, 2.022485495525404, 2.023871789886524, 2.017959712301697, 2.0163657535334507, 2.0172287997508063, 2.417949168429929, 2.0147960091021573, 2.017045349680633, 2.014543337562942, 2.0156880593562496, 2.008885833487311, 2.0085702497506723, 2.0074545319941013, 2.0029316365001986, 2.0000608222721796, 1.9982205012363368, 2.411724394838268, 1.9953786910450548, 1.9912198079616952, 1.9922663042492244, 1.9890396888522621, 1.9904259832133822, 1.9871301463473778, 1.9877405210255177, 1.9874278250186836, 1.98031290314291, 1.980451346080994, 2.404269674888903, 1.9812451708293075, 1.9806527012165012, 1.9820660976443778, 1.9801565551394933, 1.9765209201998983, 1.9727018351901293, 1.9709757427554186, 1.968221059098512, 1.9664678645970446, 1.971237122358544, 2.391102330520239, 1.9688043317098944, 1.9628903521550873, 1.960710233045654, 1.960710233045654, 1.9579376443234142, 1.9585301139362208, 1.9576670677188652, 1.9544162379849506, 1.9511204011189465, 1.9455752236744663, 2.640240771617909, 2.379828525561106, 1.9492108586140615, 1.946975579299791, 1.944813365255691, 1.9461996596168107, 1.944882586724733, 1.9477064918506817, 1.9438695017755796, 1.9465728690287776, 1.9465728690287776, 1.947959163389897, 2.3718505574679782, 1.9469126671023678, 1.9450994481332822, 1.9505754041087202, 1.951351323791701, 1.9528358436586, 1.9521562475114178, 1.953542541872538, 1.9524960455850087, 1.9524960455850087, 1.9472393498104115, 2.3505168908120613, 1.947579147884002, 1.947579147884002, 1.950691534679833, 1.9460897238931543, 1.9399520768358682, 1.9435184903064218, 1.9470849037769744, 1.944582891659284, 1.946492434164168, 1.945629387946813, 2.3426952707223503, 1.9458128380169863, 1.9412851528232988, 1.94240087057987, 1.9511406970917744, 1.9506866704170518, 1.9465970088027338, 1.9487771279121673, 1.9514804951653648, 1.9475921886865541, 1.941253186493761, 2.3461053361894866, 1.938820395845112, 1.9383842742357218, 1.9348048598237844, 1.9342816116800199, 1.936249276682542, 1.9319069435290093, 1.9324301916727737, 1.929485251851764, 1.9277591594170531, 1.9234860477325615, 2.342809499323482, 1.9248723420936817, 2.3339684850129983, 2.3205145609444324, 2.3102872522728286, 2.2997201455276346, 2.606786256880859, 2.3016296880325187, 2.2966077587318035, 2.291925627504679, 2.2840750033782307, 2.267526597579167, 2.2566085937889784, 2.2545606083460097, 2.246173620710248, 2.246880318924186, 2.246017272706831, 2.5930796612730767, 2.248973311499245, 2.2564280314486087, 2.249680009713183, 2.2440767097710657, 2.239578028614115, 2.2357589436043455, 2.2316000605209863, 2.2223742410048213, 2.219699877788361, 2.2174505372098854, 2.571918345715931, 2.2089764230397497, 2.2027245473813313, 2.1941720147407295, 2.1856015770347943, 2.181241338815928, 2.181666361453914, 2.1839157020323885, 2.1754144857954962, 2.169415281676294, 2.170669442370949, 2.555691832925125, 2.164093771734293, 2.156639051784929, 2.1509975508046506, 2.1415640668813603, 2.1346505001410936, 2.1301697240494764, 2.1127582720330573, 2.1126890505640152, 2.118853799688058, 2.1161954395669507, 2.5265975564069403, 2.1110079652613947, 2.107188880251626, 2.10594102882859, 2.1003847524127064, 2.099521706195351, 2.097428713620293, 2.0979519617640574, 2.0972273584847856, 2.098090404702141, 2.0904793367493606, 2.5175730920262827, 2.08503238481066, 2.0823469226227953, 2.081892895948073, 2.0793729787650483, 2.084918156209528, 2.0836703047864917, 2.0778032343337545, 2.0741675993941597, 2.0740112513907425, 2.068535295415305, 2.4929595915997154, 2.0676722491979493, 2.0641750571964383, 2.0663551763058714, 2.060539422256843, 2.0633120109790832, 2.053406595315737, 2.055857291029719, 2.057836055003645, 2.062265514691554, 2.0647854318745775, 2.6753342522552956, 2.4735935899463226, 2.0656484780919326, 2.0690135364269793, 2.072038796688435, 2.076313810342907, 2.08204243785756, 2.0805869220273983, 2.0803342504881828, 2.078608158053472, 2.075070748619656, 2.0715222402144358, 2.4631136097355033, 2.065655169761699, 2.054875608909593, 2.0573955260926176, 2.06273934840159, 2.0594435115355854, 2.062372448261242, 2.057419740429569, 2.051534764911499, 2.049003748757071, 2.0442794981276613, 2.462521140122696, 2.043514677416085, 2.04265163119873, 2.042128383054965, 2.0413345583066516, 2.043174879342494, 2.039217351394641, 2.0406215508210943, 2.040690772290136, 2.0370551373505412, 2.036192091133186, 2.448841646581671, 2.032129531585625, 2.0313536119026447, 2.033176027873154, 2.026767804211319, 2.033882726087092, 2.0286773467162034, 2.0266985827422763, 2.0296546215346902, 2.0274924074905902]