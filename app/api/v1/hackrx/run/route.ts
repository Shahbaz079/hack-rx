import { NextRequest, NextResponse } from "next/server";
import pdf from "pdf-parse";
import OpenAI from "openai";
import { askOpenRouter } from "@/utils/ai";

// Validate environment variables
const OPENAI_API_KEY=process.env.OPENAI_API_KEY;


// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: OPENAI_API_KEY
});

export async function POST(req: NextRequest) {
  try {
    const { documents, questions } = await req.json();

    // Validate URL format and parameters
    if (!documents || !questions || !Array.isArray(questions) || questions.length === 0) {
      return NextResponse.json(
        { error: "Invalid request. Please provide documents URL and questions array." },
        { status: 400 }
      );
    }

   

    let fullText = "";

    // 1. Download the PDF
    try {
      console.log("🔽 Fetching PDF from Azure Blob Storage:", documents);
      const pdfRes = await fetch(documents, {
        headers: {
          'Accept': 'application/pdf',
          'Cache-Control': 'no-cache'
        }
      });

      if (!pdfRes.ok) {
        const errorText = await pdfRes.text().catch(() => 'No error details available');
        throw new Error(`Failed to download document. Status: ${pdfRes.status}, Details: ${errorText}`);
      }

      const contentType = pdfRes.headers.get('content-type');
      if (!contentType?.includes('application/pdf')) {
        throw new Error(`Invalid content type: ${contentType}. Expected PDF document.`);
      }

      const pdfBuffer = await pdfRes.arrayBuffer();
      if (!pdfBuffer || pdfBuffer.byteLength === 0) {
        throw new Error('Received empty PDF buffer');
      }

      const parsed = await pdf(Buffer.from(pdfBuffer));
      fullText = parsed.text;
      
      if (!fullText.trim()) {
        throw new Error('PDF parsing succeeded but extracted text is empty');
      }
      
      console.log("✅ PDF parsed successfully, extracted", fullText.length, "characters");
    } catch (error) {
      console.error("❌ Error fetching or parsing PDF:", error);
      return NextResponse.json(
        { error: `Failed to fetch or parse the PDF document: ${(error as Error).message}` },
        { status: 500 }
      );
    }

    // 2. Chunk the document
    let chunks: string[] = [];
    try {
      if (!fullText.trim()) {
        throw new Error("Extracted PDF text is empty");
      }
      chunks = chunkText(fullText, 500); // ~500 words
      console.log("✅ Document chunked:", chunks.length, "chunks");
    } catch (error) {
      console.error("❌ Error during chunking:", error);
      return NextResponse.json(
        { error: "Failed to process the document content." },
        { status: 500 }
      );
    }

    // 3. Create embeddings for chunks
    let chunkEmbeddings: number[][] = [];
    try {
      chunkEmbeddings = await getEmbeddings(chunks);
      console.log("✅ Embeddings created for chunks");
    } catch (error) {
      console.error("❌ Error generating embeddings for chunks:", error);
      return NextResponse.json(
        { error: "Failed to generate document embeddings." },
        { status: 500 }
      );
    }

    const results: string[] = [];

    for (const question of questions) {
      try {
        if (!question.trim()) {
          results.push("Error: Empty question provided");
          continue;
        }

        console.log("🔍 Processing question:", question);

        // 4. Create embedding for question
        const [questionEmbedding] = await getEmbeddings([question]);
        console.log("✅ Question embedding generated");

        // 5. Find most relevant chunks
        const relevantChunks = findRelevantChunks(questionEmbedding, chunkEmbeddings, chunks);
        console.log("✅ Relevant chunks found:", relevantChunks.length);

        // 6. Ask OpenRouter
        const answer = await askOpenRouter(question, relevantChunks);
        console.log("✅ Answer received:", answer);

        results.push(answer);
      } catch (error) {
        console.error(`❌ Error processing question "${question}":`, error);
        results.push(`Error: Failed to process the question: ${question}`);
      }
    }

    return NextResponse.json({ answers: results });

  } catch (err: any) {
    console.error("🚨 Top-level error:", err);
    return NextResponse.json({ error: err.message || "Internal server error" }, { status: 500 });
  }
}

// 🧩 Split document into ~500-word chunks
function chunkText(text: string, maxWords: number): string[] {
  if (!text || typeof text !== 'string') {
    throw new Error('Invalid input: text must be a non-empty string');
  }
  if (!maxWords || maxWords <= 0) {
    throw new Error('Invalid input: maxWords must be a positive number');
  }

  const words = text.split(/\s+/);
  const chunks = [];

  for (let i = 0; i < words.length; i += maxWords) {
    chunks.push(words.slice(i, i + maxWords).join(" "));
  }

  return chunks;
}

// 📊 Cosine similarity
function cosineSimilarity(a: number[], b: number[]): number {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length === 0 || b.length === 0) {
    throw new Error('Invalid input: vectors must be non-empty arrays');
  }
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same length');
  }

  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

  if (magA === 0 || magB === 0) {
    throw new Error('Invalid vectors: magnitude cannot be zero');
  }

  return dot / (magA * magB);
}

// 🔍 Get top 3 relevant chunks
function findRelevantChunks(
  queryEmbedding: number[],
  docEmbeddings: number[][],
  docChunks: string[],
  topK: number = 3
): string[] {
  if (!Array.isArray(queryEmbedding) || queryEmbedding.length === 0) {
    throw new Error('Invalid query embedding');
  }
  if (!Array.isArray(docEmbeddings) || docEmbeddings.length === 0) {
    throw new Error('Invalid document embeddings');
  }
  if (!Array.isArray(docChunks) || docChunks.length === 0) {
    throw new Error('Invalid document chunks');
  }
  if (docEmbeddings.length !== docChunks.length) {
    throw new Error('Number of embeddings must match number of chunks');
  }
  if (topK <= 0 || topK > docChunks.length) {
    topK = Math.min(3, docChunks.length);
  }

  const similarities = docEmbeddings.map((embedding, index) => ({
    index,
    score: cosineSimilarity(queryEmbedding, embedding),
  }));

  const topChunks = similarities
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ index }) => docChunks[index]);

  return topChunks;
}

// 🧠 Get embeddings from OpenAI
async function getEmbeddings(texts: string[]): Promise<number[][]> {
  try {
    if (!texts || !Array.isArray(texts) || texts.length === 0) {
      throw new Error('Invalid input: texts must be a non-empty array of strings');
    }

    // Rate limiting: Process in batches of 10 texts
    const batchSize = 10;
    const embeddings: number[][] = [];
    
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const res = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: batch,
      });
      
      embeddings.push(...res.data.map((d) => d.embedding));
      
      // Add a small delay between batches to avoid rate limits
      if (i + batchSize < texts.length) {
        await new Promise(resolve => setTimeout(resolve, 200));
      }
    }
    
    return embeddings;
  } catch (error: any) {
    console.error("❌ OpenAI API Error:", error.message);
    throw new Error("Failed to generate embeddings");
  }
}
