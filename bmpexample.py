import pySALESetup as pss

mesh1 = pss.MeshfromBMP('example.bmp')

mesh1.viewMats()

mesh2 = pss.MeshfromPSSFILE()

mesh2.viewMats()
mesh2.viewVels()
