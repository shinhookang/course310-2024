from OpenGL.GL import *
from glfw.GLFW import *
import numpy as np 
import glm

from OpenGL.GL.shaders import compileProgram,compileShader


# Base position
basePos = glm.vec3(3.0, 0., 0.)
baseAng = 0.

# gBox angle
gBoxAng = 40.


# Camera Position
camY = 0.
camZ = 10.
camX = 0.
cameraUp    = glm.vec3(0., 1.,  0.)
cameraPos   = glm.vec3(camX,camY,camZ)
targetPos   = glm.vec3(0., 0., 0.)


def key_callback(window, key, scancode, action, mods):
    global basePos, baseAng, gBoxAng
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_RIGHT:
                basePos.x += 0.1
            elif key==GLFW_KEY_LEFT:
                basePos.x -= 0.1
            elif key==GLFW_KEY_DOWN:
                basePos.y -= 0.1
            elif key==GLFW_KEY_UP:
                basePos.y += 0.1
            elif key==GLFW_KEY_Z:
                baseAng += 0.1
            elif key==GLFW_KEY_C:
                baseAng -= 0.1             
            elif key==GLFW_KEY_A:
                gBoxAng += 0.5
            elif key==GLFW_KEY_D:
                gBoxAng -= 0.5                 
    
vertexShaderBoxSrc = '''
#version 330 core
layout (location = 0) in vec3 vertexPosition; 
out vec3 fragmentColor;
uniform mat4 Amat; 
uniform vec3 cvec;
void main()
{
    vec4 point = vec4(vertexPosition, 1.0);
    gl_Position = Amat * point;
    fragmentColor = cvec;
}
'''
      
vertexShaderSrc = '''
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
 
fragmentShaderSrc = '''
#version 330 core
in vec3 fragmentColor;
out vec4 fragmentColorOut;
void main()
{
   fragmentColorOut = vec4(fragmentColor, 1.0); // alpha
}
'''


 
def CreateShader(vertexShaderSrc, fragmentShaderSrc):

    shader = compileProgram(
        compileShader( vertexShaderSrc, GL_VERTEX_SHADER ),
        compileShader( fragmentShaderSrc, GL_FRAGMENT_SHADER ),
    )

    return shader
  
 
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
    
    vertices = [0.,   0.,   0.,  # v0
                1.,   0.,   0.,  # v1
                0.,   1.,   0.,  # v2
                1.,   1.,   0.]  # v3
    vertices = np.array(vertices, dtype=np.float32) # 4bytes*18=72bytes

    dtype = np.dtype(np.float32).itemsize

    # Each face has two triangles
    # The order of verticies in each triangle defines the normal direction.
    triconnect = [0,1,3,  0,3,2] # face0
    triconnect = np.array(triconnect, dtype=np.uint32) # 4bytes*9=36bytes

    dtype = np.dtype(np.float32).itemsize 

    # Create the name of Vertex Array Object(VAO)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao) 

    # Generate two Vertex Buffer Object (VBO)
    vbo = glGenBuffers(1) 
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0) # the location of the vertex attribute: position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*dtype, None)

    # Generate the name of Element Buffer Object (EBO)
    ebo = glGenBuffers(1) # Error occurs if vbo < 0
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo) # GL_ARRAY_BUFFER is vertex attribute
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triconnect.nbytes, triconnect, GL_STATIC_DRAW)

    
    return vao


def DrawBox(vao,AmatLoc,Amat,cvecLoc,cvec):
    glBindVertexArray(vao)
    glUniformMatrix4fv(AmatLoc, 1, GL_FALSE, glm.value_ptr(Amat))
    glUniform3fv(cvecLoc, 1, glm.value_ptr(cvec))
    glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, None) 


# https://docs.python.org/3/tutorial/classes.html
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


    window = glfwCreateWindow(800, 600, 'Cube', None, None)
    if not window:
        glfwTerminate()
        return Exception("Creating Window Failed!")
    
    # Make the window's context current
    glfwMakeContextCurrent(window)

    # register key callbacks, https://www.glfw.org/docs/3.3/input_guide.html
    glfwSetKeyCallback(window, key_callback)
    
    # Initialize box
    vaoBox = InitializeBox()
    
    # Create Shader program
    shaderBox = CreateShader(vertexShaderBoxSrc, fragmentShaderSrc)

    # Get the location of uniform variable 'Amat' matrix in the shader program 
    AmatLocBox = glGetUniformLocation(shaderBox, 'Amat')
    cvecLocBox = glGetUniformLocation(shaderBox, 'cvec')
    

    # uncomment this call to draw in wireframe polygons
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    
    # - - - - Define Nodes for yellow, green, pink boxes - - - 
    
    # Yellow Box
    yBox = Node(None)
    yBox.SetColor(glm.vec3(1,1,0))
    yBox.SetLocalG(glm.translate(glm.vec3(-1., 0., 0.))*glm.scale(glm.vec3(2., 1., 1.)))
    yBox.SetLocalM(glm.translate(glm.vec3(3., 0., 0.))) # from {0} to {1}

    # Green Box
    gBox = Node(yBox)
    gBox.SetColor(glm.vec3(0,1,0))
    gBox.SetLocalG(glm.translate(glm.vec3(-1./2., 0., 0.))*glm.scale(glm.vec3(1., 2., 1.)))
    gBox.SetLocalM(glm.translate(glm.vec3(0., 1., 0.))*glm.rotate(glm.radians(40), glm.vec3(0., 0., 1.))) # from {1} to {2}
    
    # Pink Box
    pBox = Node(gBox)
    pBox.SetColor(glm.vec3(255./255., 192./255., 203./255.))
    pBox.SetLocalG(glm.translate(glm.vec3(0., -1./2, 0.))*glm.scale(glm.vec3(3., 0.8, 1.)))
    pBox.SetLocalM(glm.translate(glm.vec3(0., 2., 0.))*glm.rotate(glm.radians(-20.), glm.vec3(0., 0., 1.))) # from {2} to {3}

    # Loop until the user closes the window
    while not glfwWindowShouldClose(window):

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # projection matrix 
        aspect = 800./600.
        near = 1.
        far = 10.
        fovy = glm.radians(45)
        P = glm.perspective(fovy, aspect, near, far)
        
        # view matrix
        V = glm.lookAt(cameraPos, targetPos, cameraUp)
        
        # - - - - Box - origin - - - - 
        # model matrix
        # https://glm.g-truc.net/0.9.4/api/a00206.html
        M = glm.mat4()
        S = glm.scale(glm.vec3(1.0, 1.0, 1.0))
        T = glm.translate(glm.vec3(-1./2., -1.2, 0.))
        M = T*S

        # MVP matrix 
        Amat = P*V*M
        
        # Draw Box
        cvec = glm.vec3(1,1,1) # white
        glUseProgram(shaderBox)
        DrawBox(vaoBox,AmatLocBox,Amat,cvecLocBox,cvec)


        # - - - - Update Local Coordinates for Yellow, Green, Pink Boxes - - - - 
        M1 = glm.translate(glm.vec3(basePos.x, basePos.y, basePos.z))*glm.rotate(baseAng, glm.vec3(0., 0., 1.))
        yBox.SetLocalM(M1)
        M2 = glm.translate(glm.vec3(0., 1., 0.))*glm.rotate(glm.radians(gBoxAng), glm.vec3(0., 0., 1.))
        gBox.SetLocalM(M2)

        yBox.SetGlobalM()     
        # gBox.SetGlobalM()     

        # - - - - Draw Boxes - - - 
        glUseProgram(shaderBox)
        for iBox in [yBox,gBox,pBox]:
            
            # model matrix
            M = iBox.GetModelMatrix()
        
            # MVP matrix 
            Amat = P*V*M
            
            # Draw Box
            DrawBox(vaoBox,AmatLocBox,Amat,cvecLocBox,iBox.GetColor())


        # Swap front and back buffers
        glfwSwapBuffers(window)

        # This will trigger the key_callback 
        glfwPollEvents()
    
    # delete buffer
    # glDeleteVertexArrays(1,vaoFrame)
    # glDeleteVertexArrays(1,vaoBox)
    
    # When you are done using GLFW, typically just before the application exits, you need to terminate GLFW.
    # This destroys any remaining windows and releases any other resources allocated by GLFW.
    glfwTerminate()


if __name__ == "__main__":
    main()
