from OpenGL.GL import *
from glfw.GLFW import *
import numpy as np 
import glm
from PIL import Image

from OpenGL.GL.shaders import compileProgram,compileShader


lat = glm.radians(30.)
lon = glm.radians(40.)
radius = 2.7
camY = radius * glm.sin(lat)
camZ = radius * glm.cos(lat) * glm.cos(lon)
camX = radius * glm.cos(lat) * glm.sin(lon)
cameraUp    = glm.vec3(0., 1.,  0.)
cameraPos   = glm.vec3(camX,camY,camZ)
targetPos   = glm.vec3(0., 0., 0.)


def key_callback(window, key, scancode, action, mods):
    global cameraPos, lat, lon, radius, targetPos
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action==GLFW_PRESS or action==GLFW_REPEAT:

            # camera frame
            wvec = glm.normalize(cameraPos - targetPos)
            uvec = glm.normalize(glm.cross(cameraUp,wvec))
            vvec = glm.cross(wvec,uvec)

            if key==GLFW_KEY_RIGHT:
                lon += glm.radians( 5.)
                cameraPos.z = radius * glm.cos(lat) * glm.cos(lon)
                cameraPos.x = radius * glm.cos(lat) * glm.sin(lon)
            elif key==GLFW_KEY_LEFT:
                lon -= glm.radians( 5.)
                cameraPos.z = radius * glm.cos(lat) * glm.cos(lon)
                cameraPos.x = radius * glm.cos(lat) * glm.sin(lon)
            elif key==GLFW_KEY_DOWN:
                lat -= glm.radians( 5.)
                if (lat < -glm.pi()/2.):
                    lat = -glm.pi()/2 + 0.01
                cameraPos.y = radius * glm.sin(lat)
                cameraPos.z = radius * glm.cos(lat) * glm.cos(lon)
                cameraPos.x = radius * glm.cos(lat) * glm.sin(lon)
            elif key==GLFW_KEY_UP:
                lat += glm.radians( 5.)
                if (lat > glm.pi()/2.):
                    lat = glm.pi()/2 - 0.01
                cameraPos.y = radius * glm.sin(lat)
                cameraPos.z = radius * glm.cos(lat) * glm.cos(lon)
                cameraPos.x = radius * glm.cos(lat) * glm.sin(lon)
                
            elif key==GLFW_KEY_W:
                radius -= 0.5
                if (radius < 1e-12):
                    radius = 1e-12
                cameraPos.y = radius * glm.sin(lat)
                cameraPos.z = radius * glm.cos(lat) * glm.cos(lon)
                cameraPos.x = radius * glm.cos(lat) * glm.sin(lon)

            elif key==GLFW_KEY_S:
                radius += 0.5
                if (radius < 1e-12):
                    radius = 1e-12
                cameraPos.y = radius * glm.sin(lat)
                cameraPos.z = radius * glm.cos(lat) * glm.cos(lon)
                cameraPos.x = radius * glm.cos(lat) * glm.sin(lon)      

            elif key==GLFW_KEY_0:
                lat = glm.radians(30.)
                lon = glm.radians(60.)
                radius = 15.0
                camY = radius * glm.sin(lat)
                camZ = radius * glm.cos(lat) * glm.cos(lon)
                camX = radius * glm.cos(lat) * glm.sin(lon)
                cameraPos   = glm.vec3(camX,camY,camZ)
                targetPos   = glm.vec3(0., 0., 0.)



vertexShaderSrc_sphere = '''
#version 330 core
layout (location = 0) in vec3 vertexPosition; 
layout (location = 1) in vec3 vertexNormal; 
out vec3 fragmentColor;
out vec3 viewpos;
out vec3 fragpos;
out vec3 normal;

uniform mat4 Amat; 
uniform mat4 Mmat; 
uniform vec3 cvec;


void main()
{
    vec4 point = vec4(vertexPosition, 1.0);
    gl_Position = Amat * point;
    
    // geometric information
    viewpos = cvec;
    fragpos = vertexPosition; // world (global) coordinates
    normal = vertexNormal;

    // Material properties
    vec3 material_color = vec3(1.0,0.75,0.8); // pink 
    
    fragmentColor = material_color;

}
'''

fragmentShaderSrc_sphere = '''
#version 330 core
in vec3 viewpos;
in vec3 fragpos;
in vec3 normal;
in vec3 fragmentColor;
out vec4 fragmentColorOut;

struct phLight { 
    vec3 color; 
    vec3 position; 
};

vec3 LightingModel(vec3 material_color)
{

    // ambient
    vec3 ambient = material_color;

    // diffuse
    vec3 diffuse = vec3(0,0,0);

    // specular
    vec3 specular = vec3(0,0,0);

    vec3 color =  ambient + diffuse + specular;
    return color;
}

void main()
{    
    vec3 color = LightingModel(fragmentColor);
    fragmentColorOut = vec4(color, 1.0);
}
'''



vertexShaderSrc_frame = '''
#version 330 core
layout (location = 0) in vec3 vertexPosition; 
layout (location = 1) in vec3 vertexColor; 
out vec3 fragmentColor;
uniform mat4 Amat; 
void main()
{
    vec4 point = vec4(vertexPosition, 1.0);
    gl_Position = Amat * point;
    fragmentColor = vertexColor;
}
'''
 


fragmentShaderSrc_frame = '''
#version 330 core
in vec3 fragmentColor;
out vec4 fragmentColorOut;
void main()
{
   fragmentColorOut = vec4(fragmentColor, 1.0); // alpha
}
'''

vertexShaderSrc_box = '''
#version 330 core
layout (location = 0) in vec3 vertexPosition; 
layout (location = 1) in vec3 vertexTexture; 
layout (location = 2) in vec3 vertexNormal; 

//out vec2 uvtex;
noperspective out vec2 uvtex;
out vec3 fragmentColor;
uniform mat4 Amat;
uniform mat4 Mmat;  
uniform vec3 cvec;
void main()
{

    vec4 point = vec4(vertexPosition, 1.0);
    gl_Position = Amat * point;

    // surface normal in 2D. 
    vec3 normal = vec3(0., 0., 1.);

    fragmentColor = cvec;

    uvtex = vec2(vertexTexture);
}
'''
     
 
fragmentShaderSrc_box = '''
#version 330 core
//in vec2 uvtex;
noperspective in vec2 uvtex;
in vec3 fragmentColor;
out vec4 fragmentColorOut;

uniform sampler2D mytex;

void main()
{
    // fragmentColorOut = vec4(fragmentColor, 1.0); // alpha
    fragmentColorOut = texture(mytex, uvtex);
    // fragmentColorOut = texture(mytex, uvtex) * vec4(fragmentColor, 1.0);
    /*int resolution = 10;
    int i =  int(floor(uvtex.x*resolution));
    int j =  int(floor(uvtex.y*resolution));
    if ( (i+j)%2 == 0 ) {
      fragmentColorOut = vec4(0.0, 0.0, 0.0, 1.0); 
    } else {
      fragmentColorOut = vec4(1.0, 1.0, 1.0, 1.0); 
    }
    */
}
'''


def CreateShader(vertexShaderSrc, fragmentShaderSrc):

    shader = compileProgram(
        compileShader( vertexShaderSrc, GL_VERTEX_SHADER ),
        compileShader( fragmentShaderSrc, GL_FRAGMENT_SHADER ),
    )

    return shader
  
# - - - - - IMPLEMENT InitializeWorldFrame - - - - 
def InitializeWorldFrame():
    """ InitializeWorldFrame
          y
          ^
          |
          |
          +-------> x
        /
      /
    z

    World frame (world space) is a coordinate system 
     that defines where objects are located in a virtual 3D world.

    We define world space with three orthonormal vectors.
    These vectors represent the axes of the world coordinate system, 
    ensuring that they are perpendicular to each other (orthogonal) and have unit length (normal). 

    The common choice for the three vectors corresponds to 
    the axes of a 3D Cartesian coordinate system:
      (1,0,0) for x-axis,
      (0,1,0) for y-axis,
      (0,0,1) for z-axis. 

    We draw these vectors to represent the world frame. 
    These vectors are typically drawn as scaled lines 
    originating from the origin and extending in their respective directions.
    
    We store the verticies of the lines and their colors in two Vertex Buffer Objects (VBOs).
    Then, we associate these VBOs with the Vertex Array Object (VAO).

    Input: 
    Output: vao
    """

    vertices = [0.,   0.,   0.,  
                10,    0.,   0.,  
                0.,   0.,   0.,  
                0.,   10.,   0.,  
                0.,   0.,   0.,  
                0.,   0.,   10.]  
    vertices = np.array(vertices, dtype=np.float32) # 4bytes*9=36bytes

    colors = [  165./255., 42./255.,  42./255., # Brown
                165./255., 42./255.,  42./255., # Brown
                138./255., 43./255., 226./255., # Purple
                138./255., 43./255., 226./255., # Purple
                0.0,  1.0, 0.0, # Green
                0.0,  1.0, 0.0] # Green
    colors = np.array(colors, dtype=np.float32) # 4bytes*9=36bytes

    dtype = np.dtype(np.float32).itemsize

    # Create the name of Vertex Array Object(VAO)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)     

    # Generate two Vertex Buffer Object (VBO)
    vbo_ids = glGenBuffers(2) 
    # Bind the first VBO, copy the vertex data to the target buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[0])
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0) # the location of the vertex attribute: position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*dtype, None)
    # Bind the second buffer, copy the color data to the target buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[1])
    glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
    glEnableVertexAttribArray(1) # the location of the vertex attribute: color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*dtype, None)

    return vao

def DrawFrame(vao,AmatLoc,Amat):
    glBindVertexArray(vao)
    glUniformMatrix4fv(AmatLoc, 1, GL_FALSE, glm.value_ptr(Amat))
    glDrawArrays(GL_LINES, 0, 6)


# import os
# os.getcwd()
# ifile = '/Users/pepc/Teaching/310-ComputerGraphics/code/hw3-solarsystem/sphere.obj'
def LoadObjfile(ifile):
    vertices = []
    texture = []
    normals = []
    triconnect = []
    trinormal = []
    tritexture = []
    with open(ifile) as fp:
        line = fp.readline()
        while line:
            if line[:2] == "v ":
                vx, vy, vz = [float(value) for value in line[2:].split()]
                vertices.append((vx, vy, vz))
            if line[:2] == "vt":
                tx, ty, tz = [float(value) for value in line[2:].split()]
                texture.append((tx, ty, tz))
            if line[:2] == "vn":
                nx, ny, nz = [float(value) for value in line[2:].split()]
                normals.append((nx, ny, nz))        
            if line[:2] == "f ":
                t1, t2, t3 = [value for value in line[2:].split()]
                # index starts from 0 in Python, thus -1
                triconnect.append([int(value) for value in t1.split('/')][0] - 1) 
                triconnect.append([int(value) for value in t2.split('/')][0] - 1)
                triconnect.append([int(value) for value in t3.split('/')][0] - 1)

                tritexture.append([int(value) for value in t1.split('/')][1] - 1) 
                tritexture.append([int(value) for value in t2.split('/')][1] - 1)
                tritexture.append([int(value) for value in t3.split('/')][1] - 1)

                trinormal.append([int(value) for value in t1.split('/')][2] - 1) 
                trinormal.append([int(value) for value in t2.split('/')][2] - 1)
                trinormal.append([int(value) for value in t3.split('/')][2] - 1)
            line = fp.readline()

    vertices = np.array(vertices, dtype=np.float32) # 4bytes* nvert
    triconnect = np.array(triconnect, dtype=np.uint32) # 4bytes* ntriconnect
    texture = np.array(texture, dtype=np.float32) # 4bytes* nvert
    tritexture = np.array(tritexture, dtype=np.uint32) # 4bytes* ntriconnect
    normals = np.array(normals, dtype=np.float32) # 4bytes* nvert
    trinormal = np.array(trinormal, dtype=np.uint32) # 4bytes* ntriconnect
    
    return vertices, triconnect, texture, tritexture, normals, trinormal

def vtoev(vert,conn):
    """ Vertex to Element Vertex
    Store vertices for each triangular element.
    """
    tri = []
    for t in range(0, len(conn),3):
        tri.append(vert[conn[t]])
        tri.append(vert[conn[t+1]])
        tri.append(vert[conn[t+2]])
    return np.array(tri,np.float32)

# - - - - - IMPLEMENT InitializeSphere - - - - 
def InitializeSphere():
    """ InitializeSphere
          y
          ^
          - -
       /   |   \
      |    +----|---> x
       \ /     /
       /  - -
     z

    Create a unit sphere.
    
    We load vertex coodinates and element connectivity information from sphere.obj file. 
    We store the vertex coordinates in a Vertex Buffer Objects (VBO).
    Then, we associate these VBOs with the Vertex Array Object (VAO).
    We also store the connectivty information in an Element Buffer Objects (EBO).
    
    Input: 
    Output: vao
    """

    ifile = '/Users/pepc/Teaching/310-ComputerGraphics/code/08-Texture/sphere.obj'
    vertices, triconnect,\
        texture, textureIdx,\
        normals, normalIdx = LoadObjfile(ifile)

    trivertex = vtoev(vertices,triconnect)
    tritexture = vtoev(texture,textureIdx)
    trinormal = vtoev(normals,normalIdx)
     
    
    # - - - - - - - VBO and VAO - - - - - - - 
    dtype = np.dtype(np.float32).itemsize 

    # Create the name of Vertex Array Object(VAO)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao) 

    # Generate the name of Vertex Buffer Object (VBO)
    vbo_ids = glGenBuffers(2) 
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[0])
    glBufferData(GL_ARRAY_BUFFER, trivertex.nbytes, trivertex.ravel(), GL_STATIC_DRAW)
    glEnableVertexAttribArray(0) # the location of the vertex attribute: position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*dtype, None)

    # Bind the second buffer, copy the color data to the target buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[1])
    glBufferData(GL_ARRAY_BUFFER, trinormal.nbytes, trinormal.ravel(), GL_STATIC_DRAW)
    glEnableVertexAttribArray(1) # the location of the vertex attribute: color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*dtype, None)
    
    return vao

def DrawSphere(vao,AmatLoc,Amat,MmatLoc,Mmat,cvecLoc,cvec):
    glBindVertexArray(vao)
    glUniformMatrix4fv(AmatLoc, 1, GL_FALSE, glm.value_ptr(Amat))
    glUniformMatrix4fv(MmatLoc, 1, GL_FALSE, glm.value_ptr(Mmat))
    glUniform3fv(cvecLoc, 1, glm.value_ptr(cvec))
    glDrawArrays(GL_TRIANGLES, 0, 2880) # number of triangles for the sphere. 
 


 
def InitializeBox():
    """ InitializeBox
          y
          ^
         v2------v3
          |    / |
          | /    |
         v0-----v1--> x
        /
      /
    z

    triangle 0 = [0, 1, 3]
    triangle 1 = [0, 3, 2]

    Input: 
    Output: vao
    """
    
    vertices = [[0.,   0.,   0.],  # v0
                [1.,   0.,   0.],  # v1
                [0.,   1.,   0.],  # v2
                [1.,   1.,   0.]]  # v3
    vertices = np.array(vertices, dtype=np.float32) # 4bytes*18=72bytes

    dtype = np.dtype(np.float32).itemsize

    normals = [ [0.,   0.,   1.],  # v0
                [0.,   0.,   1.],  # v1
                [0.,   0.,   1.],  # v2
                [0.,   0.,   1.]]  # v3
    normals = np.array(normals, dtype=np.float32) # 4bytes*18=72bytes

    uvtexture = [[0.,   0.,  0.],  # v0
                 [1.,   0.,  0.],  # v1
                 [0.,   1.,  0.],  # v2
                 [1.,   1.,  0.]]  # v3
    uvtexture = np.array(uvtexture, dtype=np.float32) # 4bytes*18=72bytes

    # # Each face has two triangles
    # # The order of verticies in each triangle defines the normal direction.
    triconnect = [0,1,3,  0,3,2] # face0
    triconnect = np.array(triconnect, dtype=np.uint32) # 4bytes*9=36bytes

    # dtype = np.dtype(np.float32).itemsize 

    trivertex = vtoev(vertices,triconnect)
    tritexture = vtoev(uvtexture,triconnect)
    trinormal = vtoev(normals,triconnect)
    

    # Create the name of Vertex Array Object(VAO)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao) 

    # Generate two Vertex Buffer Object (VBO)
    vbo_ids = glGenBuffers(3) 
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[0])
    glBufferData(GL_ARRAY_BUFFER, trivertex.nbytes, trivertex.ravel(), GL_STATIC_DRAW)
    glEnableVertexAttribArray(0) # the location of the vertex attribute: position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*dtype, None)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[1])
    glBufferData(GL_ARRAY_BUFFER, tritexture.nbytes, tritexture.ravel(), GL_STATIC_DRAW)
    glEnableVertexAttribArray(1) # the location of the vertex attribute: texture
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*dtype, None)
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[2])
    glBufferData(GL_ARRAY_BUFFER, trinormal.nbytes, trinormal.ravel(), GL_STATIC_DRAW)
    glEnableVertexAttribArray(2) # the location of the vertex attribute: normal
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3*dtype, None)

    # Generate the name of Element Buffer Object (EBO)
    # ebo = glGenBuffers(1) # Error occurs if vbo < 0
    # glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo) # GL_ARRAY_BUFFER is vertex attribute
    # glBufferData(GL_ELEMENT_ARRAY_BUFFER, triconnect.nbytes, triconnect, GL_STATIC_DRAW)

    
    return vao


# def DrawBox(vao,AmatLoc,Amat,cvecLoc,cvec):
#     glBindVertexArray(vao)
#     glUniformMatrix4fv(AmatLoc, 1, GL_FALSE, glm.value_ptr(Amat))
#     glUniform3fv(cvecLoc, 1, glm.value_ptr(cvec))
#     glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, None) 

def DrawBox(vao,AmatLoc,Amat,cvecLoc,cvec):
    glBindVertexArray(vao)
    glUniformMatrix4fv(AmatLoc, 1, GL_FALSE, glm.value_ptr(Amat))
    # glUniformMatrix4fv(MmatLoc, 1, GL_FALSE, glm.value_ptr(Mmat))
    glUniform3fv(cvecLoc, 1, glm.value_ptr(cvec))
    glDrawArrays(GL_TRIANGLES, 0, 6)  



# def draw_quad():
#     glBegin(GL_QUADS)
    
#     # Bottom-left vertex (corner)
#     glTexCoord2f(0.0, 0.0)
#     glVertex3f(-1.0, -1.0, 0.0)
    
#     # Bottom-right vertex (corner)
#     glTexCoord2f(1.0, 0.0)
#     glVertex3f(1.0, -1.0, 0.0)
    
#     # Top-right vertex (corner)
#     glTexCoord2f(1.0, 1.0)
#     glVertex3f(1.0, 1.0, 0.0)
    
#     # Top-left vertex (corner)
#     glTexCoord2f(0.0, 1.0)
#     glVertex3f(-1.0, 1.0, 0.0)
    
#     glEnd()

class Node:

    def __init__(self, parent):

        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        self.GlobalM = glm.mat4() # local to global transformation w.r.t. world frame
        self.LocalM = glm.mat4() # coordinate transformation w.r.t. parent coordinate system
        self.LocalG = glm.mat4() # local shape transformation
        self.color = glm.vec3(1,1,0)

    def SetColor(self,color):
        self.color = color

    def GetColor(self):
        return self.color
    
    def SetLocalM(self, M1):
        self.LocalM = M1

    def GetLocalM(self):
        return self.LocalM

    def SetLocalG(self, G1):
        self.LocalG = G1

    def GetLocalG(self):
        return self.LocalG
    
    def SetGlobalM(self):
        if self.parent is not None:
            self.GlobalM = self.parent.GetGlobalM() * self.LocalM # e.g., M = (M1*M2)*M3
        else:
            self.GlobalM = self.LocalM # e.g., M = M1

        for child in self.children:
            child.SetGlobalM()

    def GetGlobalM(self):
        return self.GlobalM
    
    def GetModelMatrix(self):
        return self.GlobalM * self.LocalG


def main():

    # Before you can use most GLFW functions, the library must be initialized.
    # On successful initialization, GLFW_TRUE is returned. If an error occurred, GLFW_FALSE is returned.
    if not glfwInit():
        return Exception("Initialization Failed!")
    
    # Ceate a window and OpenGL 3.3 context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)  
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # only use the modern OpenGL
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) #  request a forward-compatible context for mac OS


    window = glfwCreateWindow(800, 600, 'Simple Solar System', None, None)
    if not window:
        glfwTerminate()
        return Exception("Creating Window Failed!")
    
    # Make the window's context current
    glfwMakeContextCurrent(window)

    # register key callbacks, https://www.glfw.org/docs/3.3/input_guide.html
    glfwSetKeyCallback(window, key_callback)


    # - - - - - - - VAO - - - - - - - - - -
    vaoBox = InitializeBox()
    vaoSphere = InitializeSphere()
    vaoFrame = InitializeWorldFrame()
    
    # - - - - - - - Shader - - - - - - - - - -
    shader = CreateShader(vertexShaderSrc_frame, fragmentShaderSrc_frame)
    AmatLoc = glGetUniformLocation(shader, 'Amat')

    shaderSphere = CreateShader(vertexShaderSrc_sphere, fragmentShaderSrc_sphere)
    AmatLocSphere = glGetUniformLocation(shaderSphere, 'Amat')
    MmatLocSphere = glGetUniformLocation(shaderSphere, 'Mmat')
    cvecLocSphere = glGetUniformLocation(shaderSphere, 'cvec')

    shaderBox = CreateShader(vertexShaderSrc_box, fragmentShaderSrc_box)
    AmatLocBox = glGetUniformLocation(shaderBox, 'Amat')
    cvecLocBox = glGetUniformLocation(shaderBox, 'cvec')
    mytexLocBox = glGetUniformLocation(shaderBox, 'mytex')
    
    

    # uncomment this call to draw in wireframe polygons
    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    # backface culling
    glEnable(GL_CULL_FACE)
    glFrontFace(GL_CCW)

    # - - - - Define Nodes for red, green, and pink spheres - - - 
    
    # {0}
    Sphere = Node(None)
    Sphere.SetColor(glm.vec3(1,0,0))
    Sphere.SetGlobalM() 

    # backgroudn color
    glClearColor(1., 1., 1., 1.)

    # - - - - Texture - - - - 
    image = Image.open('/Users/pepc/Teaching/310-ComputerGraphics/code/08-Texture/crate.png')
    # image = Image.open('/Users/pepc/Teaching/310-ComputerGraphics/code/08-Texture/crimson1positive.png')
    # image = Image.open('/Users/pepc/Teaching/310-ComputerGraphics/code/08-Texture/RedLeavesTexture.bmp')
    
    image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip the image vertically so that (0,0) is at the bottom left.
    image_data = image.convert("RGBA").tobytes() # R,G,B, Alpha
    width, height = image.size # 256 by 256
    
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    # https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
    image.close()
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    
    glUseProgram(shaderBox)
    glUniform1i(mytexLocBox, 0)
    glActiveTexture(GL_TEXTURE0)  
    glBindTexture(GL_TEXTURE_2D, texture_id)
    

    # glUniform1i(glGetUniformLocation(shader_program, 'texture_specular'), 1)
    # glActiveTexture(GL_TEXTURE1)
    # glBindTexture(GL_TEXTURE_2D, texture_specular)
    
    
    
    

    # Loop until the user closes the window
    while not glfwWindowShouldClose(window):

        # clear color buffers to preset values
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)


        # perspective matrix
        aspect = 800./600.
        near = 0.1
        far = 100.
        fovy = glm.radians(45)
        P = glm.perspective(fovy, aspect, near, far)
        
        # view matrix
        V = glm.lookAt(cameraPos, targetPos, cameraUp)
        

        # world frame        
        M = glm.mat4()
        Amat = P*V*M
        glUseProgram(shader)
        DrawFrame(vaoFrame,AmatLoc,Amat)

        
        # - - - BOX - - - - 
        cvec = glm.vec3(1,1,0) # yellow
        glUseProgram(shaderBox)
        DrawBox(vaoBox,AmatLocBox,Amat,cvecLocBox,cvec)

        # draw_quad()
        # - - - - IMPLEMENT moving earth and moon by adjusting transformation matrices - - - - - 
        # ctime = glfwGetTime()
        # hours = ctime*300
        # days_per_year = hours/24.

        # # Earth 
        # thetaE = 2*glm.pi()/365.0*days_per_year 
        # R = glm.rotate(thetaE, glm.vec3(0., 1., 0.))
        # T = glm.translate(glm.vec3(0., 0., 5))
        # sf = 1.0
        # Sphere.SetLocalG(glm.scale(glm.vec3(sf,sf,sf)))
        # Sphere.SetLocalM(R*T) # from {0} to {1}
        # # Sphere.SetLocalM(T) # from {0} to {1}
        # Sphere.SetGlobalM() 
        
        # - - - - Draw Spheres - - - 
        # model matrix
        # M = Sphere.GetModelMatrix()
 
    
        # # MVP matrix 
        # Amat = P*V*M
        
        # Draw 
        # glUseProgram(shaderSphere)
        # DrawSphere(vaoSphere,
        #            AmatLocSphere,Amat,
        #            MmatLocSphere,M,
        #            cvecLocSphere,cameraPos)
        
        
        
        # backface culling
        glCullFace(GL_BACK)

        # Swap front and back buffers
        glfwSwapBuffers(window)

        # This will trigger the key_callback 
        glfwPollEvents()
    
    # delete buffer
    # glDeleteBuffers(1,vb # type: ignoreo)
    glDeleteVertexArrays(1,vaoSphere)
    glDeleteVertexArrays(1,vaoFrame)
    
    # When you are done using GLFW, typically just before the application exits, you need to terminate GLFW.
    # This destroys any remaining windows and releases any other resources allocated by GLFW.
    glfwTerminate()

if __name__ == "__main__":
    main()
