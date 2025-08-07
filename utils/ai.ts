// Validate OpenRouter API key
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
if (!OPENROUTER_API_KEY) {
  throw new Error('OPENROUTER_API_KEY environment variable is not set');
}

async function askOpenRouter(question: string, contextChunks: string[]): Promise<string> {
  try {
    if (!question.trim()) {
      throw new Error('Question cannot be empty');
    }
    if (!Array.isArray(contextChunks) || contextChunks.length === 0) {
      throw new Error('Context chunks must be a non-empty array');
    }

    const context = contextChunks.join("\n\n");

    const body = {
      model: "tngtech/deepseek-r1t2-chimera:free",
      messages: [
        {
          role: "system",
          content: "You are a helpful assistant that answers based only on the given context.",
        },
        {
          role: "user",
          content: `Answer the following question using the provided context.\n\nContext:\n${context}\n\nQuestion: ${question}`,
        },
      ],
    };

    const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-site.com",
        "X-Title": "HackRx QA System",
      },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(
        `OpenRouter API error: ${res.status} ${res.statusText}${errorData.error ? ` - ${errorData.error}` : ''}`
      );
    }

    const json = await res.json();
    const answer = json.choices?.[0]?.message?.content?.trim();

    if (!answer) {
      throw new Error('No answer received from OpenRouter API');
    }

    return answer;
  } catch (error: any) {
    console.error('❌ OpenRouter API Error:', error);
    throw new Error(`Failed to get answer: ${error.message}`);
  }
}

// Validate OpenAI API key
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  throw new Error('OPENAI_API_KEY environment variable is not set');
}

async function askOpenAI(question: string, contextChunks: string[]): Promise<string> {
  try {
    if (!question.trim()) {
      throw new Error('Question cannot be empty');
    }
    if (!Array.isArray(contextChunks) || contextChunks.length === 0) {
      throw new Error('Context chunks must be a non-empty array');
    }

    const context = contextChunks.join("\n\n");

    const body = {
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "system",
          content: "You are a helpful assistant that answers based only on the given context.",
        },
        {
          role: "user",
          content: `Answer the following question using the provided context.\n\nContext:\n${context}\n\nQuestion: ${question}`,
        },
      ],
    };

    const res = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(
        `OpenAI API error: ${res.status} ${res.statusText}${errorData.error ? ` - ${errorData.error}` : ''}`
      );
    }

    const json = await res.json();
    const answer = json.choices?.[0]?.message?.content?.trim();

    if (!answer) {
      throw new Error('No answer received from OpenAI API');
    }

    return answer;
  } catch (error: any) {
    console.error('❌ OpenAI API Error:', error);
    throw new Error(`Failed to get answer: ${error.message}`);
  }
}

export { askOpenAI as askOpenRouter };
