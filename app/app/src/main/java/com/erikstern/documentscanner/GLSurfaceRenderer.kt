package com.erikstern.documentscanner

import android.annotation.SuppressLint
import android.content.Context
import android.content.res.AssetManager
import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.view.MotionEvent
import android.view.Surface
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class GLSurfaceRenderer : GLSurfaceView.Renderer {
    lateinit var context: Context

    var surfaceTextureId = -1
    lateinit var surfaceTexture: SurfaceTexture
    lateinit var surface: Surface
    var nativeContext: Long = 0

    val lock = Any()
    var frameAvailable = false

    private external fun nativeCreate()
    private external fun nativeDestroy()
    private external fun nativePreInit(preview_width: Int, preview_height: Int) : IntArray
    private external fun nativeInit(assetManager: AssetManager, surface: Surface)
    private external fun nativeMotionEvent(event: Int, x: Float, y: Float)

    private external fun nativeRender()

    @SuppressLint("ClickableViewAccessibility")
    fun setSurfaceView(surfaceView: GLSurfaceView) {
        surfaceView.setEGLContextClientVersion(3)
        surfaceView.setRenderer(this)
        surfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        context = surfaceView.context

        surfaceView.setOnTouchListener { _, motionEvent ->
            when(motionEvent.action) {
                MotionEvent.ACTION_DOWN -> nativeMotionEvent(motionEvent.action, motionEvent.x, motionEvent.y)
            }
            true
        }

        nativeCreate()
    }

    fun destroyRender() {
        nativeDestroy()
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        surfaceTextureId = textures[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, surfaceTextureId)

        surfaceTexture = SurfaceTexture(surfaceTextureId)
        surfaceTexture.setOnFrameAvailableListener {
            synchronized(lock) {
                frameAvailable = true
            }
        }
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        surface = Surface(surfaceTexture)
        val dimens = nativePreInit(width, height)
        surfaceTexture.setDefaultBufferSize(dimens[0], dimens[1])
        
        nativeInit(context.assets, surface)
    }

    override fun onDrawFrame(gl: GL10?) {
        synchronized(lock) {
            if (frameAvailable) {
                surfaceTexture.updateTexImage()
                frameAvailable = false
            }
        }

        nativeRender()
    }
}