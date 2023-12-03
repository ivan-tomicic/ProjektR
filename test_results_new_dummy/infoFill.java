package ProjektR.test_results_new_dummy;
import com.fasterxml.jackson.databind.DatabindException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import org.json.simple.*;
import org.json.simple.parser.*;

import java.io.*;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

public class infoFill {

    public static void main(String[] args) throws IOException, ParseException {
        mistral1();
        mistral2();
        zephyrA();
        zephyrB();
        llama();
    }

    public static void mistral1() throws IOException, ParseException {
        JSONParser parser = new JSONParser();
        Object obj = parser.parse(new FileReader("ProjektR/test_results_new_dummy/answers/mistral-7b-instruct-v0.1.Q5_K_M.gguf.json"));
        JSONArray jsonArray = (JSONArray) obj;
        Random random=new Random();

        for (Object o : jsonArray) {
            JSONObject jsonObject = (JSONObject) o;
            JSONObject evaluation= (JSONObject) jsonObject.get("evaluation");
            JSONObject human_eval=(JSONObject) evaluation.get("human_eval");
            JSONObject fluency=(JSONObject) human_eval.get("fluency");
            JSONObject coherence=(JSONObject) human_eval.get("coherence");
            JSONObject relevance=(JSONObject) human_eval.get("relevance");
            JSONObject context_understanding=(JSONObject) human_eval.get("context_understanding");
            JSONObject overall_quality=(JSONObject) human_eval.get("overall_quality");

            overall_quality.replaceAll((k, v) -> (random.nextInt(5) + 1));
            coherence.replaceAll((k, v) -> (random.nextInt(5) + 1));
            fluency.replaceAll((k, v) -> (random.nextInt(5) + 1));
            context_understanding.replaceAll((k, v) -> (random.nextInt(5) + 1));
            relevance.replaceAll((k, v) -> (random.nextInt(5) + 1));

        }

        try (FileWriter fileWriter = new FileWriter("ProjektR/test_results_new_dummy/answers/mistral-7b-instruct-v0.1.Q5_K_M.gguf.json")) {
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(fileWriter, jsonArray);
            System.out.println("JSON file updated successfully!");
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }

    }
    public static void mistral2() throws IOException, ParseException {
        JSONParser parser = new JSONParser();
        Object obj = parser.parse(new FileReader("ProjektR/test_results_new_dummy/answers/mistral-7b-openorca.Q4_0.gguf.json"));
        JSONArray jsonArray = (JSONArray) obj;
        Random random=new Random();

        for (Object o : jsonArray) {
            JSONObject jsonObject = (JSONObject) o;
            JSONObject evaluation= (JSONObject) jsonObject.get("evaluation");
            JSONObject human_eval=(JSONObject) evaluation.get("human_eval");
            JSONObject fluency=(JSONObject) human_eval.get("fluency");
            JSONObject coherence=(JSONObject) human_eval.get("coherence");
            JSONObject relevance=(JSONObject) human_eval.get("relevance");
            JSONObject context_understanding=(JSONObject) human_eval.get("context_understanding");
            JSONObject overall_quality=(JSONObject) human_eval.get("overall_quality");

            overall_quality.replaceAll((k, v) -> (random.nextInt(5) + 1));
            coherence.replaceAll((k, v) -> (random.nextInt(5) + 1));
            fluency.replaceAll((k, v) -> (random.nextInt(5) + 1));
            context_understanding.replaceAll((k, v) -> (random.nextInt(5) + 1));
            relevance.replaceAll((k, v) -> (random.nextInt(5) + 1));

        }

        try (FileWriter fileWriter = new FileWriter("ProjektR/test_results_new_dummy/answers/mistral-7b-openorca.Q4_0.gguf.json")) {
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(fileWriter, jsonArray);
            System.out.println("JSON file updated successfully!");
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }
    public static void zephyrA() throws IOException, ParseException {
        JSONParser parser = new JSONParser();
        Object obj = parser.parse(new FileReader("ProjektR/test_results_new_dummy/answers/zephyr-7b-alpha.Q5_K_M.gguf.json"));
        JSONArray jsonArray = (JSONArray) obj;
        Random random=new Random();

        for (Object o : jsonArray) {
            JSONObject jsonObject = (JSONObject) o;
            JSONObject evaluation= (JSONObject) jsonObject.get("evaluation");
            JSONObject human_eval=(JSONObject) evaluation.get("human_eval");
            JSONObject fluency=(JSONObject) human_eval.get("fluency");
            JSONObject coherence=(JSONObject) human_eval.get("coherence");
            JSONObject relevance=(JSONObject) human_eval.get("relevance");
            JSONObject context_understanding=(JSONObject) human_eval.get("context_understanding");
            JSONObject overall_quality=(JSONObject) human_eval.get("overall_quality");

            overall_quality.replaceAll((k, v) -> (random.nextInt(5) + 1));
            coherence.replaceAll((k, v) -> (random.nextInt(5) + 1));
            fluency.replaceAll((k, v) -> (random.nextInt(5) + 1));
            context_understanding.replaceAll((k, v) -> (random.nextInt(5) + 1));
            relevance.replaceAll((k, v) -> (random.nextInt(5) + 1));

        }

        try (FileWriter fileWriter = new FileWriter("ProjektR/test_results_new_dummy/answers/zephyr-7b-alpha.Q5_K_M.gguf.json")) {
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(fileWriter, jsonArray);
            System.out.println("JSON file updated successfully!");
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }
    public static void zephyrB() throws IOException, ParseException {
        JSONParser parser = new JSONParser();
        Object obj = parser.parse(new FileReader("ProjektR/test_results_new_dummy/answers/zephyr-7b-beta.Q5_K_M.gguf.json"));
        JSONArray jsonArray = (JSONArray) obj;
        Random random=new Random();

        for (Object o : jsonArray) {
            JSONObject jsonObject = (JSONObject) o;
            JSONObject evaluation= (JSONObject) jsonObject.get("evaluation");
            JSONObject human_eval=(JSONObject) evaluation.get("human_eval");
            JSONObject fluency=(JSONObject) human_eval.get("fluency");
            JSONObject coherence=(JSONObject) human_eval.get("coherence");
            JSONObject relevance=(JSONObject) human_eval.get("relevance");
            JSONObject context_understanding=(JSONObject) human_eval.get("context_understanding");
            JSONObject overall_quality=(JSONObject) human_eval.get("overall_quality");

            overall_quality.replaceAll((k, v) -> (random.nextInt(5) + 1));
            coherence.replaceAll((k, v) -> (random.nextInt(5) + 1));
            fluency.replaceAll((k, v) -> (random.nextInt(5) + 1));
            context_understanding.replaceAll((k, v) -> (random.nextInt(5) + 1));
            relevance.replaceAll((k, v) -> (random.nextInt(5) + 1));

        }

        try (FileWriter fileWriter = new FileWriter("ProjektR/test_results_new_dummy/answers/zephyr-7b-beta.Q5_K_M.gguf.json")) {
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(fileWriter, jsonArray);
            System.out.println("JSON file updated successfully!");
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }
    public static void llama() throws IOException, ParseException {
            JSONParser parser = new JSONParser();
            Object obj = parser.parse(new FileReader("ProjektR/test_results_new_dummy/answers/llama-2-7b-chat.Q5_K_M.gguf.json"));
            JSONArray jsonArray = (JSONArray) obj;
            Random random=new Random();

            for (Object o : jsonArray) {
                JSONObject jsonObject = (JSONObject) o;
                JSONObject evaluation= (JSONObject) jsonObject.get("evaluation");
                JSONObject human_eval=(JSONObject) evaluation.get("human_eval");
                JSONObject fluency=(JSONObject) human_eval.get("fluency");
                JSONObject coherence=(JSONObject) human_eval.get("coherence");
                JSONObject relevance=(JSONObject) human_eval.get("relevance");
                JSONObject context_understanding=(JSONObject) human_eval.get("context_understanding");
                JSONObject overall_quality=(JSONObject) human_eval.get("overall_quality");

                overall_quality.replaceAll((k, v) -> (random.nextInt(5) + 1));
                coherence.replaceAll((k, v) -> (random.nextInt(5) + 1));
                fluency.replaceAll((k, v) -> (random.nextInt(5) + 1));
                context_understanding.replaceAll((k, v) -> (random.nextInt(5) + 1));
                relevance.replaceAll((k, v) -> (random.nextInt(5) + 1));

            }

        try (FileWriter fileWriter = new FileWriter("ProjektR/test_results_new_dummy/answers/llama-2-7b-chat.Q5_K_M.gguf.json")) {
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(fileWriter, jsonArray);
            System.out.println("JSON file updated successfully!");
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }

    }
}
