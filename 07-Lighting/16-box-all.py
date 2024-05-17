from OpenGL.GL import *
from glfw.GLFW import *
import numpy as np 
import glm

from OpenGL.GL.shaders import compileProgram,compileShader


# Base position
basePos = glm.vec3(0.0, 0., 0.)
baseAng = 0.

# Camera Position
camY = 0.
camZ = 10.
camX = 0.
cameraUp    = glm.vec3(0., 1.,  0.)
cameraPos   = glm.vec3(camX,camY,camZ)
targetPos   = glm.vec3(0., 0., 0.)

def key_callback(window, key, scancode, action, mods):
    global basePos, baseAng
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

vertexShaderSrc_poin = '''
#version 330 core
layout (location = 0) in vec3 vertexPosition; 
out vec3 fragmentColor;
uniform vec3 pos;
void main()
{
    // gl_Position = vec4(vertexPosition, 1.0); // w
    gl_Position = vec4(pos, 1.0); // w
    fragmentColor = vec3(1.0, 0.0, 0.0);
}
'''


vertexShaderSrc = '''
#version 330 core
layout (location = 0) in vec3 vertexPosition; 
layout (location = 1) in vec3 vertexNormal; 

out vec3 fragmentColor;
uniform mat4 Amat;
uniform mat4 Mmat;  
uniform vec3 vpos;
uniform vec3 lpos;
void main()
{

    vec4 point = vec4(vertexPosition, 1.0);
    gl_Position = Amat * point;
 
    // ambient color
    vec3 rho_a = 0.1 * vec3(1.0,0.75,0.8); // pink
    vec3 I_a = vec3(1.,1.,1.); // white
    vec3 ambient = rho_a * I_a;

    // diffuse color 
    vec3 normal = vec3(0., 0., 1.); // surface normal in 2D. 
    vec3 mypos = vec3(Mmat * point);
    vec3 ellvec = normalize(lpos - mypos);
    float costheta = dot(ellvec,normal);
    vec3 rho_d = vec3(1.0,0.75,0.8); // pink
    vec3 I_d = vec3(1., 1., 1.); // white
    vec3 diffuse = rho_d * costheta* I_d;

    // specular color
    vec3 vvec = normalize(vpos - mypos); // view vector
    // vec3 rvec = reflect(-ellvec, normal); // r vector 
    float ldotn = dot(ellvec, normal); // l dot n 
    vec3 rvec = 2.0*ldotn*normal - ellvec; 
    float specExponent = 100.;
    float specFactor = pow( max(dot(rvec, vvec), 0.0), specExponent);
    vec3 I_s = vec3(1.,1.,1.); // white
    vec3 rho_s = vec3(1.,1.,1.); // white
    // vec3 rho_s = vec3(0.,0.,0.); // black
    vec3 specular = specFactor * rho_s * I_s;
    
    fragmentColor = specular + diffuse + ambient;
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
  

def InitializePoint():
    """ InitializePoint
          y
          ^
          | 
          |    /  p(x,y,z)
          | /     
          +---------> x
        /
      /
    z

    point 

    Input: 
    Output: vao
    """
    
    vertices = [0.,   0.,   0.]  # v3
    vertices = np.array(vertices, dtype=np.float32) # 4bytes*18=72bytes
    
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
    
    return vao


def DrawPoint(vao,
              posLoc,pos):
    glBindVertexArray(vao)
    glUniform3fv(posLoc, 1, glm.value_ptr(pos))
    glPointSize(20)
    glDrawArrays(GL_POINTS, 0, 3) 



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


def DrawBox(vao,
            AmatLoc,Amat,
            MmatLoc,Mmat,
            vposLoc,vpos,
            lposLoc,lpos):
    glBindVertexArray(vao)
    glUniformMatrix4fv(AmatLoc, 1, GL_FALSE, glm.value_ptr(Amat))
    glUniformMatrix4fv(MmatLoc, 1, GL_FALSE, glm.value_ptr(Mmat))
    glUniform3fv(vposLoc, 1, glm.value_ptr(vpos))
    glUniform3fv(lposLoc, 1, glm.value_ptr(lpos))
    glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, None) 


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
    
    # Initialize box and point
    vaoBox = InitializeBox()
    vaoPoint = InitializePoint()
    
    # Create Shader program
    shaderBox = CreateShader(vertexShaderSrc, fragmentShaderSrc)
    shaderPoint = CreateShader(vertexShaderSrc_poin, fragmentShaderSrc)

    # Get the location of uniform variable 'Amat' matrix in the shader program 
    AmatLocBox = glGetUniformLocation(shaderBox, 'Amat')
    MmatLocBox = glGetUniformLocation(shaderBox, 'Mmat')
    vposLocBox = glGetUniformLocation(shaderBox, 'vpos')
    lposLocBox = glGetUniformLocation(shaderBox, 'lpos')
    posLocPoint = glGetUniformLocation(shaderPoint, 'pos')
    

    # uncomment this call to draw in wireframe polygons
    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    
    # Loop until the user closes the window
    while not glfwWindowShouldClose(window):

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # projection matrix 
        aspect = 800./600.
        near = 0.01
        far = 100.
        fovy = glm.radians(45)
        P = glm.perspective(fovy, aspect, near, far)
        
        # view matrix
        V = glm.lookAt(cameraPos, targetPos, cameraUp)
        
        
        # Transformation of coordinate system from {0} to {1}
        M1 = glm.translate(glm.vec3(basePos.x, basePos.y, basePos.z))*glm.rotate(baseAng, glm.vec3(0., 0., 1.))
        # Local Shape Transformation in {1}
        S = glm.scale(glm.vec3(2., 1., 1.))
        T = glm.translate(glm.vec3(-1., 0., 0.))
        G1 = T * S

        # model matrix
        # M = glm.mat4()
        M = M1 * G1
        Mmat = M
        
        # MVP matrix 
        Amat = P*V*M
        
        # Draw Box
        lightPos = glm.vec3(-0.9,0.1,1.0)
        glUseProgram(shaderBox)
        DrawBox(vaoBox,
                AmatLocBox,Amat,
                MmatLocBox,Mmat,
                vposLocBox,cameraPos,
                lposLocBox,lightPos)


        # Draw Point
        # glUseProgram(shaderPoint)
        # DrawPoint(vaoPoint,
        #           posLocPoint,lightPos)

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
