package com.zeroagent.app

import android.app.Activity
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.*
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : Activity() {
    private val ink = Color.rgb(5, 5, 5)
    private val paper = Color.rgb(247, 247, 244)
    private val muted = Color.rgb(113, 113, 109)
    private lateinit var feed: LinearLayout
    private lateinit var composer: EditText
    private lateinit var taskCount: TextView
    private val prefs by lazy { getSharedPreferences("zero_agent", MODE_PRIVATE) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = paper
        window.navigationBarColor = paper
        window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR
        setContentView(buildScreen())
    }

    private fun buildScreen(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(paper)
            setPadding(dp(20), dp(12), dp(20), dp(16))
        }
        root.addView(header())
        root.addView(space(16))
        root.addView(statusCard())
        root.addView(space(20))
        root.addView(label("TODAY'S QUEUE", 11, muted, 1.2f))
        root.addView(space(8))
        root.addView(queueRow("Distill the brief", "Ready to begin", true))
        root.addView(space(7))
        root.addView(queueRow("Find a next step", "Waiting for direction", false))
        root.addView(space(18))
        root.addView(label("WORKSPACE", 11, muted, 1.2f))
        root.addView(space(8))

        val scroll = ScrollView(this).apply { isFillViewport = true; overScrollMode = View.OVER_SCROLL_NEVER }
        feed = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        addGreeting()
        scroll.addView(feed)
        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        root.addView(composer())
        return root
    }

    private fun header(): View {
        val row = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        val icon = ImageView(this).apply { setImageResource(com.zeroagent.app.R.drawable.app_mark) }
        row.addView(icon, LinearLayout.LayoutParams(dp(38), dp(38)))
        val names = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(10), 0, 0, 0) }
        names.addView(label("ZERO AGENT", 17, ink, 1.0f, Typeface.BOLD))
        names.addView(label("Personal operating system", 12, muted, 1.0f))
        row.addView(names, LinearLayout.LayoutParams(0, -2, 1f))
        val dot = TextView(this).apply {
            text = "●  ONLINE"; setTextColor(ink); textSize = 10f; letterSpacing = .08f
            setPadding(dp(10), dp(7), dp(8), dp(7)); background = rounded(Color.WHITE, Color.rgb(215,215,210), 99)
        }
        row.addView(dot)
        return row
    }

    private fun statusCard(): View {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; setPadding(dp(18), dp(17), dp(18), dp(16))
            background = rounded(ink, ink, 16)
        }
        card.addView(label("YOUR AGENT IS CLEAR", 11, Color.rgb(190,190,185), 1.25f))
        card.addView(space(8))
        card.addView(label("What needs to move?", 25, paper, 1.0f, Typeface.BOLD))
        card.addView(space(5))
        card.addView(label("Give Zero a thought, a plan, or a loose end.", 14, Color.rgb(211,211,207), 1.1f))
        return card
    }

    private fun queueRow(title: String, sub: String, done: Boolean): View {
        val row = LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL; setPadding(dp(13), dp(10), dp(13), dp(10))
            background = rounded(Color.WHITE, Color.rgb(222,222,217), 12)
        }
        val tick = TextView(this).apply {
            text = if (done) "✓" else ""; gravity = Gravity.CENTER; textSize = 15f
            setTextColor(if (done) paper else ink)
            background = oval(if (done) ink else paper, ink)
        }
        row.addView(tick, LinearLayout.LayoutParams(dp(23), dp(23)))
        val copy = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(10), 0, 0, 0) }
        copy.addView(label(title, 14, ink, 1f, Typeface.BOLD))
        copy.addView(label(sub, 12, muted, 1f))
        row.addView(copy, LinearLayout.LayoutParams(0, -2, 1f))
        row.setOnClickListener { Toast.makeText(this, if (done) "Marked as active" else "Added to your active queue", Toast.LENGTH_SHORT).show() }
        return row
    }

    private fun composer(): View {
        val box = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL; setPadding(dp(13), dp(8), dp(8), dp(8)); background = rounded(Color.WHITE, Color.rgb(195,195,190), 14) }
        composer = EditText(this).apply {
            hint = "Message Zero…"; setHintTextColor(muted); setTextColor(ink); textSize = 15f
            backgroundColor = Color.TRANSPARENT; maxLines = 3; imeOptions = EditorInfo.IME_ACTION_SEND
            setSingleLine(false)
            setOnEditorActionListener { _, id, _ -> if (id == EditorInfo.IME_ACTION_SEND) { sendMessage(); true } else false }
        }
        box.addView(composer, LinearLayout.LayoutParams(0, -2, 1f))
        val send = TextView(this).apply {
            text = "↑"; gravity = Gravity.CENTER; textSize = 22f; setTextColor(paper)
            background = oval(ink, ink); contentDescription = "Send message"; isClickable = true
            setOnClickListener { sendMessage() }
        }
        box.addView(send, LinearLayout.LayoutParams(dp(39), dp(39)))
        return box
    }

    private fun addGreeting() {
        val hour = SimpleDateFormat("H", Locale.getDefault()).format(Date()).toInt()
        val greeting = when (hour) { in 5..11 -> "Good morning."; in 12..17 -> "Good afternoon."; else -> "Good evening." }
        addBubble("ZERO", "$greeting I’m here to turn the fuzzy thing into a small, clear next move.", false)
        addSuggestion("Plan my day")
        addSuggestion("Capture a thought")
    }

    private fun sendMessage() {
        val text = composer.text.toString().trim()
        if (text.isEmpty()) return
        composer.setText("")
        addBubble("YOU", text, true)
        val response = replyTo(text)
        feed.postDelayed({ addBubble("ZERO", response, false) }, 260)
    }

    private fun replyTo(input: String): String {
        val lower = input.lowercase(Locale.getDefault())
        return when {
            "plan" in lower || "day" in lower -> "A simple plan: choose one meaningful outcome, protect a 45-minute block for it, then clear one small loose end. What is the outcome?"
            "help" in lower -> "I can help you shape a plan, capture an idea, or find the smallest next action. Start wherever the thought is messiest."
            "idea" in lower || "thought" in lower -> "Captured. Give the idea a working title, then answer: who is it for, and what changes if it works?"
            else -> "Noted. The smallest useful next move is to name what ‘done’ looks like, then take the first action that makes it more real."
        }
    }

    private fun addBubble(who: String, message: String, mine: Boolean) {
        val wrap = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(4), 0, dp(9)); gravity = if (mine) Gravity.END else Gravity.START }
        wrap.addView(label(who, 10, muted, 1.2f))
        wrap.addView(space(4))
        val bubble = TextView(this).apply {
            text = message; textSize = 15f; setTextColor(if (mine) paper else ink); setLineSpacing(dp(2).toFloat(), 1f)
            setPadding(dp(14), dp(11), dp(14), dp(11)); background = rounded(if (mine) ink else Color.WHITE, if (mine) ink else Color.rgb(220,220,215), 13)
        }
        wrap.addView(bubble, LinearLayout.LayoutParams(if (mine) dp(280) else dp(300), -2))
        feed.addView(wrap)
    }

    private fun addSuggestion(text: String) {
        val chip = TextView(this).apply {
            this.text = text; textSize = 13f; setTextColor(ink); setPadding(dp(12), dp(8), dp(12), dp(8))
            background = rounded(Color.TRANSPARENT, Color.rgb(185,185,180), 99)
            setOnClickListener { composer.setText(text); composer.requestFocus() }
        }
        feed.addView(chip, LinearLayout.LayoutParams(-2, -2).apply { bottomMargin = dp(7) })
    }

    private fun label(text: String, size: Int, color: Int, spacing: Float, style: Int = Typeface.NORMAL) = TextView(this).apply {
        this.text = text; textSize = size.toFloat(); setTextColor(color); letterSpacing = spacing - 1f; typeface = Typeface.create("sans", style)
    }
    private fun space(h: Int) = Space(this).apply { layoutParams = LinearLayout.LayoutParams(1, dp(h)) }
    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()
    private fun rounded(fill: Int, stroke: Int, radius: Int) = GradientDrawable().apply { setColor(fill); cornerRadius = dp(radius).toFloat(); setStroke(dp(1), stroke) }
    private fun oval(fill: Int, stroke: Int) = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(fill); setStroke(dp(1), stroke) }
}
