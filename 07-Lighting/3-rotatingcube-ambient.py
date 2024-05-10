from OpenGL.GL import *
from glfw.GLFW import *
import numpy as np 
import glm

from OpenGL.GL.shaders import compileProgram,compileShader


def key_callback(window, key, scancode, action, mods):
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
     
vertexShaderSrc = '''
#version 330 core
layout (location = 0) in vec3 vertexPosition; 
layout (location = 1) in vec3 vertexNormal; 

out vec3 fragmentColor;
uniform mat4 Amat; 
void main()
{
    vec4 point = vec4(vertexPosition, 1.0);
    gl_Position = Amat * point;
    // fragmentColor = vec3(1.0,0.75,0.8); // pink


    // ambient color
    float rho_a = 0.2;
    vec3 color_a = vec3(0.0,0.5,0.0); // green
    vec3 ambient = rho_a * color_a;

    fragmentColor = ambient;
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
 
 

def createVAOCube():
    # - - - - - - - VBO and VAO - - - - - - - 
    # Triangle vertices for cube
    vertices = [-1., -1.,  1., # v0
                 1., -1.,  1., # v1
                -1.,  1.,  1., # v2
                 1.,  1.,  1., # v3
                -1.,  1., -1., # v4
                 1.,  1., -1., # v5
                -1., -1., -1., # v6
                 1., -1., -1.] # v7
    vertices = 1./2.*np.array(vertices, dtype=np.float32) # 4bytes*24=96bytes

    # Define averaged normal vector at each vertex
    normals = [ -1., -1.,  1., # v0
                 1., -1.,  1., # v1
                -1.,  1.,  1., # v2
                 1.,  1.,  1., # v3
                -1.,  1., -1., # v4
                 1.,  1., -1., # v5
                -1., -1., -1., # v6
                 1., -1., -1.] # v7
    normals = 1./np.sqrt(3) * np.array(normals, dtype=np.float32) # 4bytes*9=36bytes

    # Each face has two triangles
    # The order of verticies in each triangle defines the normal direction.
    triconnect = [0,1,2,  2,1,3, # face0
                  2,3,4,  3,5,4, # face1
                  4,5,6,  6,5,7, # face2
                  0,6,7,  1,0,7, # face3
                  1,7,3,  3,7,5, # face4
                  6,0,4,  4,0,2] # face5
    triconnect = np.array(triconnect, dtype=np.uint32) # 4bytes*9=36bytes

    dtype = np.dtype(np.float32).itemsize 
    
    # Create the name of Vertex Array Object(VAO)
    # Bind a vertex array object to the name of VAO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao) 

    # Generate the name of Vertex Buffer Object (VBO)
    # Bind a buffer object to the name of VBO
    vbo = glGenBuffers(2) # Error occurs if vbo < 0
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]) # GL_ARRAY_BUFFER is vertex attribute
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0) # the location of the vertex attribute: position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*dtype, None)

    # Bind the second buffer, copy the color data to the target buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
    glEnableVertexAttribArray(1) # the location of the vertex attribute: color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*dtype, None)

    
    # Generate the name of Element Buffer Object (EBO)
    # Bind a buffer object to the name of EBO
    ebo = glGenBuffers(1) # Error occurs if vbo < 0
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo) # GL_ARRAY_BUFFER is vertex attribute
    # Copy the connectivity data into the target buffer 
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triconnect.nbytes, triconnect, GL_STATIC_DRAW)

    return vao


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
    


    # - - - - - - - VAO - - - - - - - - - -
    vaoCube = createVAOCube()
    
    # - - - - - - - Shader - - - - - - - - - -
    # Note that VAO should be assigned 
    shader = CreateShader(vertexShaderSrc, fragmentShaderSrc)

    # Get the location of uniform variable 'Amat' matrix in the shader program 
    AmatLoc = glGetUniformLocation(shader, 'Amat')
    

    # uncomment this call to draw in wireframe polygons
    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    # backface culling
    # glEnable(GL_CULL_FACE)
    glFrontFace(GL_CCW)
    
    # Loop until the user closes the window
    while not glfwWindowShouldClose(window):

        # clear color buffers to preset values
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # glEnable(GL_DEPTH_TEST)

        glUseProgram(shader)

        # - - - - - - Viewing Control - - - - - 
        # perspective 
        # https://glm.g-truc.net/0.9.9/api/a00665.html#ga747c8cf99458663dd7ad1bb3a2f07787
        aspect = 800./600.
        near = 1.
        far = 10.
        fovy = glm.radians(45)
        P = glm.perspective(fovy, aspect, near, far)
        
        # view matrix
        ctime = glfwGetTime()
        radius = 3.
        camX = glm.sin(ctime)*radius
        camZ = glm.cos(ctime)*radius
        camY = 3.0
        cameraUp    = glm.vec3(0., 1.,  0.)
        cameraPos   = glm.vec3(camX,camY,camZ)
        targetPos   = glm.vec3(0., 0., 0.)
        V = glm.lookAt(cameraPos, targetPos, cameraUp)
        

        # Define a 3D affine transfomation matrix (model matrix)
        M = glm.mat4()

        # MVP matrix 
        Amat = P*V*M
        
        # draw cube
        glUseProgram(shader)
        glBindVertexArray(vaoCube)
        glUniformMatrix4fv(AmatLoc, 1, GL_FALSE, glm.value_ptr(Amat)) # GLM matrix is column-major.
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None) 
        
        # backface culling
        glCullFace(GL_BACK)

        # Swap front and back buffers
        glfwSwapBuffers(window)

        # This will trigger the key_callback 
        glfwPollEvents()
    
    # delete buffer
    # glDeleteBuffers(1,vb # type: ignoreo)
    glDeleteVertexArrays(1,vaoCube)
    
    # When you are done using GLFW, typically just before the application exits, you need to terminate GLFW.
    # This destroys any remaining windows and releases any other resources allocated by GLFW.
    glfwTerminate()

if __name__ == "__main__":
    main()
