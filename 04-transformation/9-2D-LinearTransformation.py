from OpenGL.GL import *
from glfw.GLFW import *
import numpy as np 

from OpenGL.GL.shaders import compileProgram,compileShader

vertexShaderSrc = '''
#version 330 core
layout (location = 0) in vec3 vertexPosition; 
out vec3 fragmentColor;
uniform mat2 Amat;  // 2x2 matrix
void main()
{
    gl_Position = vec4(vertexPosition, 1.0); // w
    gl_Position.xy = Amat * vertexPosition.xy;
    fragmentColor = vec3(0.0, 1.0, 0.0);
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

    window = glfwCreateWindow(640, 640, 'Hello World! a Triangle', None, None)
    if not window:
        glfwTerminate()
        return Exception("Creating Window Failed!")
    
    # Make the window's context current
    glfwMakeContextCurrent(window)

    
    # - - - - - - - VBO and VAO - - - - - - - 
    # Triangle vertices 
    vertices = [ 0.0, 0.0, 0.0, # v0
                 1/2, 0.0, 0.0, # v1
                 1/4, 1/4, 0.0] # v2
    vertices = np.array(vertices, dtype=np.float32) # 4bytes*9=36bytes
     
    
    # Create the name of Vertex Array Object(VAO)
    # Bind a vertex array object to the name of VAO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao) 

    # Generate the name of Vertex Buffer Object (VBO)
    # Bind a buffer object to the name of VBO
    vbo = glGenBuffers(1) # Error occurs if vbo < 0
    glBindBuffer(GL_ARRAY_BUFFER, vbo) # GL_ARRAY_BUFFER is vertex attribute

    # Copy the vertex data into the target buffer 
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)


    # Specify vertex attribute
    valoc = 0 # the location of the vertex attribute
    glVertexAttribPointer(valoc, 3, GL_FLOAT, GL_FALSE, 3*np.dtype(np.float32).itemsize, None) # vertex data consists of 3 floating points numbers
    glEnableVertexAttribArray(valoc) 
    
    # - - - - - - - Shader - - - - - - - - - -
    # Note that VAO should be assigned 
    shader = CreateShader(vertexShaderSrc, fragmentShaderSrc)
 
    # Get the location of uniform variable 'Amat' matrix in the shader program 
    AmatLoc = glGetUniformLocation(shader, 'Amat')

    # Define a 2D linear transfomation matrix
    Amat = np.identity(2)

    linearLransformation = 'identity'
    # linearLransformation = 'scaling'
    # linearLransformation = 'rotation'
    # linearLransformation = 'reflectionx'
    # linearLransformation = 'reflectiony'
    # linearLransformation = 'shearx'
    # linearLransformation = 'sheary'

    if linearLransformation == 'scaling':
        Amat[0,0] = 2 # stretch an object along the x-axis.
        Amat[1,1] = 1/2 # compress the object along the y-axis.
    elif linearLransformation == 'rotation':
        Amat[0,0] = np.cos(np.pi/18) 
        Amat[0,1] =-np.sin(np.pi/18) 
        Amat[1,0] = np.sin(np.pi/18) 
        Amat[1,1] = np.cos(np.pi/18) 
    elif linearLransformation == 'reflectionx':
        Amat[1,1] = -1 #reflection about x axis
    elif linearLransformation == 'reflectiony':
        Amat[0,0] = -1 # reflection about y axis
    elif linearLransformation == 'shearx':
        Amat[0,1] = 1 # horizontal shear
    elif linearLransformation == 'sheary':
        Amat[1,0] = 1 # vertical shear
        

    # Pass Amat to the shader program as parameters.
    # Here, GL_TRUE is set to convert a row-major Numpy matrix to a column-major matrix.
    glUseProgram(shader) # don't forget to activate the program object before updating the uniform variable.
    glUniformMatrix2fv(AmatLoc, 1, GL_TRUE, Amat)

    # Loop until the user closes the window
    while not glfwWindowShouldClose(window):

        # clear color buffers to preset values
        glClear(GL_COLOR_BUFFER_BIT) 

        # - - - - - - triangle - - - - - 
        # Draw a triangle with three verticies from array data by starting with the first vertex
        glUseProgram(shader)
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, 3) 
        # - - - - - - triangle - - - - - 

        # Swap front and back buffers
        glfwSwapBuffers(window)

        # Poll for and process events 
        glfwPollEvents()
    
    # delete buffer
    glDeleteBuffers(1,vbo)
    glDeleteVertexArrays(1,vao)
    
    # When you are done using GLFW, typically just before the application exits, you need to terminate GLFW.
    # This destroys any remaining windows and releases any other resources allocated by GLFW.
    glfwTerminate()

if __name__ == "__main__":
    main()
