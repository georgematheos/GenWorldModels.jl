### Conditional entropy of ground truth relations given inferred ###

### GenWorldModels split/merge: ###
times_gwm = [164.928, 1731.714, 172.088, 179.136, 187.246, 194.181, 11.038, 201.203, 208.247, 215.36, 222.595, 229.953, 237.017, 244.333, 251.578, 258.713, 265.778, 17.45, 272.833, 280.738, 289.882, 298.811, 305.997, 313.375, 320.927, 328.002, 335.055, 342.183, 23.98, 349.161, 356.216, 363.095, 370.053, 377.044, 383.883, 390.838, 397.693, 404.521, 411.343, 30.827, 418.413, 425.192, 432.194, 439.13, 445.95, 452.918, 459.839, 466.655, 473.296, 480.119, 37.609, 487.054, 493.874, 500.602, 507.278, 514.568, 521.215, 527.804, 534.597, 541.38, 548.175, 44.7, 554.878, 561.555, 568.163, 575.215, 582.109, 589.042, 595.828, 602.495, 609.392, 616.446, 52.105, 623.328, 630.151, 637.094, 644.511, 651.636, 658.723, 665.74, 672.833, 679.83, 686.65, 0.0, 59.135, 693.463, 700.316, 706.957, 713.505, 720.212, 727.165, 733.972, 740.884, 747.743, 754.568, 66.152, 761.404, 768.317, 775.111, 781.857, 788.813, 795.767, 802.705, 809.532, 816.497, 823.351, 72.971, 830.133, 837.014, 843.86, 850.862, 857.791, 864.559, 871.56, 878.275, 885.176, 892.077, 80.001, 899.051, 906.077, 912.906, 919.858, 926.803, 933.482, 940.184, 946.978, 954.161, 960.848, 86.973, 967.548, 974.211, 981.078, 987.929, 994.897, 1001.847, 1008.969, 1015.854, 1022.742, 1029.65, 93.878, 1036.558, 1043.87, 1050.846, 1057.775, 1064.986, 1072.001, 1078.806, 1085.838, 1092.96, 1099.92, 100.852, 1106.954, 1113.989, 1120.833, 1127.73, 1134.618, 1141.583, 1148.436, 1155.252, 1162.224, 1169.025, 108.184, 1175.931, 1182.809, 1189.906, 1196.863, 1203.968, 1211.212, 1218.123, 1225.111, 1232.249, 1239.377, 115.057, 1246.311, 1253.242, 1260.399, 1267.638, 1274.834, 1283.015, 1290.118, 1297.593, 1304.586, 1311.543, 122.247, 1318.573, 1325.432, 1332.418, 1339.361, 1346.42, 1353.472, 1360.705, 1367.709, 1374.561, 1381.545, 5.192, 129.24, 1388.654, 1395.679, 1402.65, 1409.538, 1416.525, 1423.657, 1430.394, 1437.578, 1444.361, 1451.217, 136.469, 1457.927, 1464.619, 1471.455, 1478.635, 1485.432, 1492.5, 1499.31, 1506.255, 1513.207, 1520.457, 143.424, 1527.615, 1534.513, 1541.546, 1548.536, 1555.437, 1562.457, 1569.214, 1576.13, 1583.1, 1590.106, 150.62, 1596.956, 1603.878, 1610.771, 1617.577, 1624.244, 1631.101, 1637.678, 1644.38, 1650.967, 1657.488, 157.622, 1664.415, 1671.141, 1678.131, 1684.75, 1691.337, 1697.995, 1704.653, 1711.439, 1718.148, 1724.883]
ents_gwm = Any[1.2136541610520057, 0.8974923326622507, 1.2058741533658572, 1.1961231715023593, 1.1956899859160195, 1.1727527827822806, 1.8083144900266108, 1.178862986070313, 1.1631642672296947, 1.1628670778767036, 1.1510375811608364, 1.140233196571156, 1.110281642243528, 1.1125751153098091, 1.119558013414046, 1.1227791060498, 1.1217546677261372, 1.7041165971272416, 1.1127119323280896, 1.0929249501604905, 1.0899986139125797, 1.0783449868157757, 1.0857570473206488, 1.0711741160398147, 1.0754977290563852, 1.0600698330015488, 1.0621377541316048, 1.0552188332719552, 1.6236007551616025, 1.0591436310130655, 1.0570521705209148, 1.051185409452554, 1.0437467914601526, 1.0422127028741939, 1.0437415676592334, 1.0347029834624693, 1.0390329699302046, 1.0417614621340163, 1.0413274314817669, 1.560891461161489, 1.0326362707796137, 1.0373798948517647, 1.0366217767644932, 1.0301506666086149, 1.0471100731569496, 1.0534273834828238, 1.0521684975574914, 1.0503438167931254, 1.0448000466360163, 1.0410863522555318, 1.5286092711310832, 1.0406455956066625, 1.0413596903974685, 1.0217540093314048, 1.0263011558520199, 1.024520356379314, 1.037442908969587, 1.0286069410033576, 1.020215289860228, 1.026764108722198, 1.0272581515099335, 1.514835753949388, 1.0216140261392237, 1.0182324818961317, 1.0101709222028992, 1.0140123283944786, 1.0170003701274115, 1.0060572317580556, 1.0055673949526667, 1.0049877076488112, 1.0029026678217516, 1.007909880578179, 1.4665960306446275, 1.0121325142559354, 1.0157969927167407, 1.0102779254704266, 1.0167981144924731, 1.0170798142507105, 1.007523274244272, 1.0016598651362159, 1.0039438357517136, 1.005832853435443, 1.0114617226765796, 2.1853653601409353, 1.4377750922099426, 1.0073860932211327, 1.0098640909435708, 1.0081464824487858, 1.0025161948640753, 0.9990714745870106, 1.006752188556019, 0.9986217908235979, 0.9896396832116197, 0.9872862304619368, 0.9904665034177915, 1.3987857062159836, 0.9827019634231452, 0.9854824591353518, 0.97833421008091, 0.9738238844542387, 0.9770434174450878, 0.9737606237164439, 0.971954669895193, 0.9678352700554336, 0.9695744631068521, 0.9737046986646319, 1.3661444056769065, 0.9777529113709229, 0.9689385157277391, 0.9684356364078361, 0.9662178994138014, 0.965215720460061, 0.9885756595215925, 0.991482120110322, 0.9902386696105578, 0.9805772407910888, 0.9687113947968087, 1.3364776627489696, 0.9699263914594978, 0.9708342448672439, 0.9721541436870385, 0.9712143458906748, 0.9627709933412822, 0.955784818131887, 0.9536565585896708, 0.9438275723136889, 0.9462800313912031, 0.9432139868413717, 1.318462613349203, 0.9570056746691372, 0.9571006761383285, 0.9464844693252684, 0.9563245257506326, 0.940948230197214, 0.9383469065944818, 0.943361209281314, 0.9402531122932151, 0.9469489061391193, 0.9414634590989208, 1.3040198012191544, 0.9410833281261869, 0.9430976089884128, 0.9462525042405265, 0.9389890592269255, 0.9267654966845295, 0.9304537664055145, 0.9535846952288917, 0.9535746758114462, 0.9546526120918956, 0.9549951207855032, 1.2823036490576174, 0.9466418926648179, 0.9475236846755783, 0.9420796319653362, 0.9438528390945055, 0.9443133969234663, 0.9382986921391011, 0.9366706879615496, 0.9389480468841713, 0.9397402149136949, 0.9313983488341869, 1.2661968403919104, 0.9322253554930423, 0.9261696760681034, 0.9315699450667537, 0.9263999505046839, 0.9194504153595883, 0.9224341820654713, 0.9198098771206432, 0.9153829329370602, 0.9244201440041677, 0.9028057533216448, 1.2608615417735287, 0.9055547155529936, 0.8973136256204144, 0.8926985783323261, 0.8895703769479202, 0.8883746550931528, 0.889840882967337, 0.8858805871621671, 0.8863860055854664, 0.8891724584621975, 0.8866972816957903, 1.2473652355635285, 0.8879731373933789, 0.8874369869518468, 0.8764995020106249, 0.8615325487210327, 0.8624711876798427, 0.8593601072717364, 0.8598019104798992, 0.8632087173567902, 0.8627133015578567, 0.8561119401541085, 1.9637060644635997, 1.25573282227757, 0.855799014517075, 0.8579983843722321, 0.8616849124098027, 0.8613737517663171, 0.8585314140757451, 0.8637701027748773, 0.8721518064464442, 0.8709087682399351, 0.8737791055166862, 0.8726309401708213, 1.2491230402544906, 0.8655850266963095, 0.878048026333835, 0.8853189646305857, 0.8857933741534995, 0.8912584030063175, 0.8964904596662305, 0.8983296169414703, 0.8970997984771445, 0.8891062668381208, 0.8923440579091834, 1.2308398280697903, 0.8801849646076766, 0.8788593123390155, 0.881764520595491, 0.8759557354437898, 0.8748919497649731, 0.8710130951673289, 0.8738580205789146, 0.8733630489448613, 0.8738232549832645, 0.8798094885418679, 1.2209637952911894, 0.8830579529062714, 0.8892020828767844, 0.8878870271925645, 0.8900907892536436, 0.898016104441183, 0.8984308038623411, 0.9039036175906973, 0.9022767762827577, 0.895680199085086, 0.8952017111175165, 1.216424230281089, 0.8903433801181265, 0.881144766920215, 0.8898800356394497, 0.8868259830512094, 0.8850989480018809, 0.8853120402776722, 0.8846988425824208, 0.8856450814862519, 0.8848927695715866, 0.8881220028261856]
### Vanilla Gen split/merge: ###
times_van_sm = [993.261, 999.988, 1004.566, 73.896, 1011.351, 1017.696, 1022.333, 1027.8, 1033.707, 1039.389, 1046.828, 1053.435, 1060.113, 1069.687, 75.55, 1076.664, 1081.547, 1087.712, 1095.448, 1101.188, 1108.953, 1116.03, 1120.509, 1125.457, 1131.651, 82.135, 1137.435, 1144.882, 1150.927, 1157.495, 1165.323, 1172.803, 1179.763, 1189.312, 1195.033, 1201.208, 2.645, 88.993, 1208.322, 1210.652, 1219.905, 1227.339, 1234.9, 1242.82, 1248.94, 1256.567, 1264.069, 1269.766, 94.626, 1278.024, 1287.273, 1296.048, 1303.525, 1310.913, 1318.149, 1322.28, 1327.53, 1336.048, 1341.285, 100.45, 1348.254, 1353.494, 1360.398, 1365.23, 1370.518, 1377.057, 1384.924, 1390.92, 1395.823, 1402.009, 107.123, 1409.851, 1416.567, 1425.409, 1431.353, 1439.381, 1445.762, 1450.415, 1456.772, 1463.176, 1471.206, 115.572, 1477.66, 1481.584, 1485.483, 1494.328, 1499.434, 1505.327, 1510.827, 1518.449, 1523.238, 1529.597, 118.294, 1537.511, 122.775, 128.937, 135.888, 142.247, 5.698, 147.93, 152.893, 158.152, 162.948, 168.414, 173.834, 180.614, 185.732, 192.074, 197.971, 8.821, 203.641, 208.474, 214.094, 218.9, 224.728, 229.545, 236.291, 241.878, 248.618, 254.14, 11.162, 260.085, 266.764, 274.583, 280.169, 285.382, 290.896, 296.873, 302.456, 308.817, 317.466, 16.218, 321.909, 326.74, 333.378, 337.761, 344.945, 351.655, 357.586, 363.092, 370.906, 375.337, 20.387, 378.963, 386.063, 393.045, 400.623, 407.487, 413.616, 417.387, 424.717, 430.911, 437.786, 25.777, 443.65, 449.092, 454.861, 458.636, 465.59, 472.177, 478.726, 486.887, 493.656, 502.695, 30.526, 506.174, 511.548, 517.299, 524.943, 529.083, 533.635, 540.06, 546.169, 552.98, 558.189, 0.0, 36.395, 565.109, 570.782, 576.626, 584.317, 589.684, 596.245, 602.784, 611.401, 613.908, 620.52, 42.461, 628.622, 633.602, 637.381, 641.531, 646.908, 654.044, 658.628, 666.779, 673.761, 678.8, 46.695, 684.182, 689.172, 694.179, 700.318, 705.706, 713.467, 720.119, 727.116, 732.966, 736.846, 51.883, 744.695, 753.453, 758.829, 765.904, 770.635, 778.183, 783.361, 788.477, 797.31, 802.466, 56.64, 810.596, 817.025, 822.215, 828.203, 833.992, 841.625, 850.176, 856.662, 861.979, 869.157, 62.664, 876.237, 883.267, 890.962, 895.91, 902.159, 905.778, 912.071, 919.279, 925.532, 933.142, 68.351, 940.263, 945.239, 953.734, 961.763, 970.612, 976.509, 984.971]
ents_van_sm = Any[0.9858603316638427, 0.9873657844064853, 0.9888803686329416, 1.746748448637063, 0.9776455734636699, 0.9813823962018448, 0.9814516176708868, 0.9813244955868262, 0.9755107579319419, 0.975929930547322, 0.9760429515060102, 0.9729634871110412, 0.9718531436819718, 0.970493005488952, 1.717614510212554, 0.9692079962122003, 0.967609258698709, 0.9644540183118887, 0.9644540183118887, 0.9634465716753884, 0.9517927801169881, 0.9503583690113541, 0.9487173867168742, 0.9463613578199009, 0.9457490811717809, 1.6996240084586085, 0.9490788261984713, 0.9611850763319243, 0.9585828968773062, 0.9578795456421862, 0.9578795456421862, 0.9597707340053908, 0.9626907253007441, 0.9631709081435668, 0.957117539752019, 0.9610637468458482, 2.3587098756573948, 1.6834445763453474, 0.9600574679906239, 0.9574512728860928, 0.9527069636502501, 0.9607091460803308, 0.9585328707720259, 0.9551458518605658, 0.9531711433977199, 0.946762919735885, 0.9458107407626777, 0.9435374721201987, 1.6700237698949985, 0.9433907780793477, 0.9415743452463414, 0.9427860604045953, 0.9391047883957024, 0.9370699183182822, 0.9370699183182822, 0.9289675685512745, 0.9289675685512745, 0.9222267375957955, 0.9228209947544188, 1.6513238433749806, 0.9222285251416121, 0.9243646228787797, 0.9237763232012176, 0.9237763232012176, 0.923560906801506, 0.923560906801506, 0.9168337524242142, 0.9227800573381351, 0.9211652381774634, 0.9217405369136411, 1.6254764325702182, 0.904917498998948, 0.9039662402679555, 0.9142586004733568, 0.9136013166931469, 0.9124488172508112, 0.914553023221437, 0.9135515340659977, 0.913662874937325, 0.9155491591765812, 0.9117259042315679, 1.6162581605757051, 0.9163989683652068, 0.9157041033113769, 0.9157041033113769, 0.9166157255222598, 0.9076642119876088, 0.913074260577363, 0.9183258405180598, 0.9294602057929033, 0.9281366982505035, 0.9283041452253245, 1.6046373293383394, 0.9295314436560895, 1.5998929880676878, 1.5889804730291637, 1.5635548885619193, 1.548053389392844, 2.2838249820325713, 1.5240130959634424, 1.514855178869351, 1.5048578140763849, 1.4905842984552602, 1.4814711528315694, 1.470538648379486, 1.4637038354076464, 1.4590681806988426, 1.4453288036898932, 1.4399798398905777, 2.2507921414805305, 1.418271309391777, 1.4143304406012316, 1.4015940833173213, 1.3973755051074188, 1.3918885266327263, 1.3802253384780796, 1.3679615649641224, 1.3526683943872593, 1.3411440816026785, 1.334129496249772, 2.197009668941905, 1.328904043570547, 1.3339609536671357, 1.3191863576124385, 1.3129966856191817, 1.314275931204577, 1.3107334310961252, 1.3043641426611554, 1.3041709800765133, 1.298346061921521, 1.2851666844510499, 2.1198819606363175, 1.2831457612136703, 1.276625203954976, 1.2759537046252298, 1.2739301819418116, 1.2716080378369532, 1.2676209117442647, 1.2625322019158371, 1.265426885655958, 1.2617598057382962, 1.2565926017491265, 2.0781816754998848, 1.252035798094538, 1.2260509460793838, 1.222305040763949, 1.2091444125942108, 1.2019161005575063, 1.2050862979215842, 1.202941822889233, 1.2123696706395706, 1.2160350970098495, 1.2159602372281542, 2.0032241526656303, 1.2162312356570213, 1.2152699639338869, 1.2175172881182181, 1.2034524047437956, 1.206232184245982, 1.207331519567842, 1.2119735477868252, 1.2067373768333827, 1.192008310287864, 1.1859648821318305, 1.9573943588902998, 1.1795924717720687, 1.1752408341586562, 1.1858132297138173, 1.1834028768108433, 1.1800415079915942, 1.1752822958443867, 1.17807031320693, 1.1765235896748703, 1.165470362305348, 1.1672442979725322, 2.4122405759670835, 1.9411992559407363, 1.1689423322691623, 1.166065168908357, 1.166655514668563, 1.1741776872318863, 1.1826816040177932, 1.1767001841613938, 1.1755835081937986, 1.1586864064882025, 1.1572052336001222, 1.158914050906708, 1.8985448123963122, 1.147650527649212, 1.144049078097864, 1.1409557641486425, 1.144299792322157, 1.147759020651902, 1.136323589833925, 1.1352098892852116, 1.1332299575300093, 1.1228404179369236, 1.1237216175405458, 1.8815283525595201, 1.1190795893215626, 1.1176240734914007, 1.116752890595442, 1.1118086034274495, 1.1074775235306515, 1.105009778603222, 1.110729101657996, 1.1059754809965658, 1.0944682401482246, 1.0957591335036316, 1.823338758129401, 1.0915283040248236, 1.0909808415441067, 1.0900783120831512, 1.072485850937888, 1.0669986815668826, 1.0661385976423279, 1.066539757733919, 1.0558623806128848, 1.045729950468242, 1.044898011013017, 1.812761263186429, 1.048591654026086, 1.0568141569798273, 1.0561672169604843, 1.0444273522248217, 1.047200675135766, 1.0473260027083018, 1.0427754252442698, 1.0393481889005047, 1.0204977876497867, 1.013168118045397, 1.7882765517353796, 1.013906782588873, 1.0114771015526483, 1.0104279691064344, 1.0053793969879763, 1.00111213532609, 1.0013023914901928, 0.9998818371778293, 0.9966009430843574, 0.9943325296592731, 0.993292839465673, 1.7635731968247945, 0.9898016306016649, 0.9980501752993423, 0.9987721424199288, 0.9975022302010411, 0.9955636836594196, 0.9932180195451455, 0.988596898545464]
### Vanilla Gen generic inference: ###
times_van_generic = [872.092, 85.449, 878.287, 884.472, 890.634, 897.023, 903.114, 909.251, 915.408, 921.685, 928.044, 934.083, 91.858, 940.141, 946.357, 952.764, 958.826, 965.015, 971.134, 977.237, 983.366, 989.497, 995.633, 98.506, 1001.897, 1008.083, 1014.08, 1020.16, 1026.15, 1032.356, 1038.447, 1044.599, 1050.64, 1056.905, 104.873, 1063.278, 1069.31, 1075.498, 1081.657, 1087.685, 1093.842, 1099.843, 1105.835, 1111.836, 1117.868, 111.263, 1124.157, 1130.132, 1136.261, 1142.314, 1148.384, 1154.616, 1160.767, 1166.926, 1172.879, 1179.097, 117.611, 1185.033, 1191.09, 1197.078, 1203.157, 1209.225, 1215.607, 1221.684, 1227.667, 1233.966, 1239.997, 6.675, 124.334, 1246.163, 1252.252, 1258.353, 1264.342, 1270.753, 1276.736, 1282.754, 1288.756, 1294.771, 1300.964, 130.661, 1306.855, 1312.856, 1318.925, 1324.947, 1331.223, 1337.085, 1343.179, 1349.289, 1355.382, 1361.746, 137.016, 1367.811, 1373.85, 1379.784, 1385.789, 1391.935, 1397.973, 1404.006, 1410.016, 1415.971, 1422.375, 143.415, 1428.238, 1434.172, 1440.069, 1446.057, 1452.18, 1458.202, 1464.119, 1469.899, 1475.908, 1482.115, 149.783, 1488.157, 1494.157, 1499.984, 1506.041, 1511.949, 1518.132, 1524.144, 1530.224, 1536.272, 1542.614, 156.339, 1548.525, 162.866, 169.395, 175.926, 181.987, 13.412, 188.266, 194.627, 201.061, 207.419, 213.696, 220.201, 226.852, 233.297, 239.688, 245.916, 20.069, 252.501, 258.768, 265.191, 271.562, 277.905, 284.198, 290.529, 296.876, 303.389, 309.672, 26.719, 315.996, 322.396, 329.002, 335.196, 341.464, 347.701, 354.034, 360.425, 366.829, 373.094, 33.368, 379.423, 385.622, 391.752, 397.938, 404.126, 410.437, 416.661, 422.922, 429.227, 435.649, 39.889, 441.686, 447.909, 454.277, 460.583, 466.825, 473.117, 479.367, 485.919, 492.189, 498.333, 46.403, 504.7, 510.859, 517.39, 523.585, 529.791, 536.102, 542.457, 548.629, 554.856, 561.045, 52.984, 567.302, 573.51, 579.729, 586.012, 592.6, 598.798, 604.953, 611.328, 617.438, 623.62, 0.0, 59.385, 629.72, 635.907, 642.177, 648.646, 654.836, 661.05, 667.342, 673.531, 679.826, 686.139, 65.762, 692.443, 698.733, 705.271, 711.48, 717.731, 723.924, 730.112, 736.132, 742.475, 748.735, 72.256, 755.167, 761.194, 767.207, 773.398, 779.681, 785.701, 792.01, 798.139, 804.321, 810.8, 79.049, 816.785, 822.927, 828.893, 835.077, 841.27, 847.415, 853.616, 859.782, 866.061]
ents_van_generic = Any[1.899516362833369, 2.2881358122784534, 1.9038236723571416, 1.8963039415253706, 1.8994487381192062, 1.8935494529841557, 1.894085100225011, 1.8958597659220315, 1.899319773356707, 1.897766899235426, 1.894121204905982, 1.8971145867449704, 2.2750020118170324, 1.9004138663232226, 1.8978117257733609, 1.8976752088858786, 1.8987993551848226, 1.9018792556088362, 1.8985550110541083, 1.900839223261206, 1.8954409761903157, 1.8952469388951467, 1.8968764772447715, 2.2697289589182805, 1.8956113049886056, 1.8969815416913025, 1.8932906591839138, 1.8951593731116838, 1.889162145509026, 1.8878718581546512, 1.8853925663695783, 1.8824773621791373, 1.8797515937912461, 1.8777665576840021, 2.263150346686134, 1.8740572163756313, 1.870322188955438, 1.8701756372830638, 1.8669585401956332, 1.868734504025215, 1.8647800804947092, 1.8636298560417823, 1.8632208630955698, 1.856956383595817, 1.8582138384734521, 2.25590206535853, 1.8604781993320603, 1.860440752404356, 1.8615007786035684, 1.860503449316081, 1.8570751604289175, 1.8542307375894895, 1.8524276460367095, 1.8472509086422908, 1.8442581765061785, 1.8500286301787001, 2.243743970768589, 1.849230630402028, 1.841363429635933, 1.836632983392757, 1.8370530163892587, 1.8347318538639952, 1.8375076852707475, 1.8369192711690214, 1.8346556024603524, 1.8296649910261986, 1.8249133626980567, 2.4943670700089333, 2.2331726512541827, 1.8290197852420569, 1.827346703448947, 1.8264049764732242, 1.826371304951255, 1.8235164573260665, 1.8266939950399836, 1.8224832792763335, 1.826723088112612, 1.82703331381198, 1.828462293807113, 2.224300036200053, 1.8272103110382085, 1.8246915086394104, 1.8321661997366778, 1.8341264510349915, 1.8345127012495102, 1.8339136743550577, 1.8380154215727544, 1.8385705175313976, 1.839724486359964, 1.8368652642946721, 2.203414406352024, 1.8365291599854805, 1.8365291599854805, 1.8394857710932433, 1.836092146843267, 1.8301571581257277, 1.833347447701042, 1.8371939863730171, 1.832685714423217, 1.8350131216816, 1.8323770124707845, 2.1970362142633215, 1.8322315206240536, 1.829226626226843, 1.8296515181504331, 1.839125155976351, 1.8411118430097757, 1.8378509411914137, 1.841124827437128, 1.843488036011338, 1.8418847617270095, 1.8351025407100072, 2.200877377289647, 1.8327817865596132, 1.833308684913076, 1.8296748561112912, 1.8281882615591487, 1.8295893159395813, 1.8265395074403583, 1.8278820656897412, 1.8250019469001793, 1.8230961374877812, 1.8193672287637177, 2.195235192287886, 1.8192134625893177, 2.185664307829085, 2.1717939659082166, 2.1610878323677083, 2.150484108664974, 2.460395072434395, 2.1520134723332456, 2.149194002372889, 2.1444303359996786, 2.139050297729758, 2.1229435998815664, 2.114956159963653, 2.1190408512049896, 2.1127503525396665, 2.1147823512509953, 2.1143778918796117, 2.4445974222993696, 2.1189099485458285, 2.126158457078742, 2.1194718304873237, 2.113351298494064, 2.1110423765652055, 2.108107858554574, 2.104981950957103, 2.096352694822573, 2.0933479389820078, 2.0896602592060494, 2.420714093084551, 2.0793378042786403, 2.07107520006315, 2.061440311905016, 2.055218867329117, 2.05249936903068, 2.0559366218029242, 2.0565732438586544, 2.049939912113074, 2.044465093308451, 2.0464363183575722, 2.4053319235322768, 2.0402727613289806, 2.0345530635037132, 2.0303878947124137, 2.0218332643049797, 2.0154926866275624, 2.012239213970191, 1.995396151469833, 1.9949053908692365, 1.9961677188153166, 1.9932380716963483, 2.3758311523545026, 1.9862571471972659, 1.9855633580051204, 1.984550987254546, 1.9773823419617258, 1.9757878224821268, 1.9743006777933798, 1.9757174078560007, 1.9729967416621164, 1.9759918706885764, 1.9715435527348524, 2.3667524047240476, 1.9657893142426763, 1.9617432488841149, 1.9603040268544478, 1.9560278508702356, 1.9597935414089283, 1.9578826634933617, 1.95241661623488, 1.94716005313273, 1.9482115745158117, 1.9407894790882827, 2.3438380400342624, 1.9360224008468885, 1.9338058919840493, 1.938069777553702, 1.9320829011564342, 1.935205432500408, 1.927110423988145, 1.9337759736512679, 1.9368186544681134, 1.9396315912221727, 1.9409624864347, 2.528319999161691, 2.322267145562597, 1.9428981412048736, 1.9433706188115587, 1.9472579997140032, 1.9491627599214334, 1.9549453673439994, 1.9537845897757882, 1.9547231708880324, 1.953135967514338, 1.9477720928719244, 1.9452023940465328, 2.31341571440025, 1.9388583652263012, 1.9266151238387412, 1.931882805685343, 1.9347977772573566, 1.9316987547535283, 1.934830553958765, 1.9294778081604087, 1.9245158102525515, 1.9224392747878318, 1.9197013120379094, 2.3138232951178805, 1.9193077242053234, 1.9187283179898011, 1.918030977800397, 1.9162454339373969, 1.9180672315277891, 1.9147372145492525, 1.9167343474136982, 1.9172856563141298, 1.9139518654049117, 1.9132769054076963, 2.2980567486669097, 1.9074326610492134, 1.905200089846123, 1.9037435477924176, 1.8978248738148793, 1.9073186072598152, 1.9008416514154582, 1.8992894703418608, 1.9007827400090653, 1.898966384996498]
