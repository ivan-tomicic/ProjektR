package ProjektR.test_results_new_dummy;

import au.com.bytecode.opencsv.CSVWriter;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class kappa {

    public static void main(String[] args) throws IOException, ParseException {
        llama();
        mistral_1();
        mistral_2();
        zephyr_1();
        zephyr_2();
    }
    public static void writeRow(CSVWriter writer, int rev1, int rev2, int rev3) {
        String[] row = new String[3];
        row[0] = String.valueOf(rev1);
        row[1] = String.valueOf(rev2);
        row[2] = String.valueOf(rev3);
        writer.writeNext(row);
    }

    public static void llama() throws IOException, ParseException {
        List<Integer> values=new ArrayList<>();
        JSONParser parser = new JSONParser();
        Object obj = parser.parse(new FileReader("ProjektR/test_results_new_dummy/answers/llama-2-7b-chat.Q5_K_M.gguf.json"));
        JSONArray jsonArray = (JSONArray) obj;


        String csv_overall_quality = "ProjektR/test_results_new_dummy/kappa_llama/reviewer_scores_overall_quality.csv";
        String csv_coherence = "ProjektR/test_results_new_dummy/kappa_llama/reviewer_scores_coherence.csv";
        String csv_fluency = "ProjektR/test_results_new_dummy/kappa_llama/reviewer_scores_fluency.csv";
        String csv_context_understanding = "ProjektR/test_results_new_dummy/kappa_llama/reviewer_scores_context_understanding.csv";
        String csv_relevance = "ProjektR/test_results_new_dummy/kappa_llama/reviewer_scores_relevance.csv";

        CSVWriter writer_overall_quality = new CSVWriter(new FileWriter(csv_overall_quality));
        CSVWriter writer_coherence = new CSVWriter(new FileWriter(csv_coherence));
        CSVWriter writer_fluency = new CSVWriter(new FileWriter(csv_fluency));
        CSVWriter writer_context_understanding = new CSVWriter(new FileWriter(csv_context_understanding));
        CSVWriter writer_relevance = new CSVWriter(new FileWriter(csv_relevance));


        String[] header = {"Reviewer_1", "Reviewer_2","Reviewer_3"};
        writer_overall_quality.writeNext(header);
        writer_coherence.writeNext(header);
        writer_fluency.writeNext(header);
        writer_context_understanding.writeNext(header);
        writer_relevance.writeNext(header);

        for (Object o : jsonArray) {
            JSONObject jsonObject = (JSONObject) o;
            JSONObject evaluation= (JSONObject) jsonObject.get("evaluation");
            JSONObject human_eval=(JSONObject) evaluation.get("human_eval");
            JSONObject fluency=(JSONObject) human_eval.get("fluency");
            JSONObject coherence=(JSONObject) human_eval.get("coherence");
            JSONObject relevance=(JSONObject) human_eval.get("relevance");
            JSONObject context_understanding=(JSONObject) human_eval.get("context_understanding");
            JSONObject overall_quality=(JSONObject) human_eval.get("overall_quality");


            for(Object value: overall_quality.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_overall_quality,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: coherence.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_coherence,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: fluency.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_fluency,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: relevance.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_relevance,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: context_understanding.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_context_understanding,values.get(0),values.get(1),values.get(2));
            values.clear();


        }

        writer_overall_quality.flush();
        writer_overall_quality.close();
        writer_coherence.flush();
        writer_coherence.close();
        writer_fluency.flush();
        writer_fluency.close();
        writer_relevance.flush();
        writer_relevance.close();
        writer_context_understanding.flush();
        writer_context_understanding.close();


    }

    public static void mistral_1() throws IOException, ParseException {
        List<Integer> values=new ArrayList<>();
        JSONParser parser = new JSONParser();
        Object obj = parser.parse(new FileReader("ProjektR/test_results_new_dummy/answers/mistral-7b-instruct-v0.1.Q5_K_M.gguf.json"));
        JSONArray jsonArray = (JSONArray) obj;


        String csv_overall_quality = "ProjektR/test_results_new_dummy/kappa_mistral_1/reviewer_scores_overall_quality.csv";
        String csv_coherence = "ProjektR/test_results_new_dummy/kappa_mistral_1/reviewer_scores_coherence.csv";
        String csv_fluency = "ProjektR/test_results_new_dummy/kappa_mistral_1/reviewer_scores_fluency.csv";
        String csv_context_understanding = "ProjektR/test_results_new_dummy/kappa_mistral_1/reviewer_scores_context_understanding.csv";
        String csv_relevance = "ProjektR/test_results_new_dummy/kappa_mistral_1/reviewer_scores_relevance.csv";

        CSVWriter writer_overall_quality = new CSVWriter(new FileWriter(csv_overall_quality));
        CSVWriter writer_coherence = new CSVWriter(new FileWriter(csv_coherence));
        CSVWriter writer_fluency = new CSVWriter(new FileWriter(csv_fluency));
        CSVWriter writer_context_understanding = new CSVWriter(new FileWriter(csv_context_understanding));
        CSVWriter writer_relevance = new CSVWriter(new FileWriter(csv_relevance));


        String[] header = {"Reviewer_1", "Reviewer_2","Reviewer_3"};
        writer_overall_quality.writeNext(header);
        writer_coherence.writeNext(header);
        writer_fluency.writeNext(header);
        writer_context_understanding.writeNext(header);
        writer_relevance.writeNext(header);

        for (Object o : jsonArray) {
            JSONObject jsonObject = (JSONObject) o;
            JSONObject evaluation= (JSONObject) jsonObject.get("evaluation");
            JSONObject human_eval=(JSONObject) evaluation.get("human_eval");
            JSONObject fluency=(JSONObject) human_eval.get("fluency");
            JSONObject coherence=(JSONObject) human_eval.get("coherence");
            JSONObject relevance=(JSONObject) human_eval.get("relevance");
            JSONObject context_understanding=(JSONObject) human_eval.get("context_understanding");
            JSONObject overall_quality=(JSONObject) human_eval.get("overall_quality");



            for(Object value: overall_quality.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_overall_quality,values.get(0),values.get(1),values.get(2));
values.clear();


            for(Object value: coherence.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_coherence,values.get(0),values.get(1),values.get(2));
values.clear();


            for(Object value: fluency.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_fluency,values.get(0),values.get(1),values.get(2));
values.clear();


            for(Object value: relevance.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_relevance,values.get(0),values.get(1),values.get(2));
values.clear();


            for(Object value: context_understanding.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_context_understanding,values.get(0),values.get(1),values.get(2));
values.clear();



        }

        writer_overall_quality.flush();
        writer_overall_quality.close();
        writer_coherence.flush();
        writer_coherence.close();
        writer_fluency.flush();
        writer_fluency.close();
        writer_relevance.flush();
        writer_relevance.close();
        writer_context_understanding.flush();
        writer_context_understanding.close();


    }

    public static void mistral_2() throws IOException, ParseException {
        List<Integer> values=new ArrayList<>();
        JSONParser parser = new JSONParser();
        Object obj = parser.parse(new FileReader("ProjektR/test_results_new_dummy/answers/mistral-7b-openorca.Q4_0.gguf.json"));
        JSONArray jsonArray = (JSONArray) obj;


        String csv_overall_quality = "ProjektR/test_results_new_dummy/kappa_mistral_2/reviewer_scores_overall_quality.csv";
        String csv_coherence = "ProjektR/test_results_new_dummy/kappa_mistral_2/reviewer_scores_coherence.csv";
        String csv_fluency = "ProjektR/test_results_new_dummy/kappa_mistral_2/reviewer_scores_fluency.csv";
        String csv_context_understanding = "ProjektR/test_results_new_dummy/kappa_mistral_2/reviewer_scores_context_understanding.csv";
        String csv_relevance = "ProjektR/test_results_new_dummy/kappa_mistral_2/reviewer_scores_relevance.csv";

        CSVWriter writer_overall_quality = new CSVWriter(new FileWriter(csv_overall_quality));
        CSVWriter writer_coherence = new CSVWriter(new FileWriter(csv_coherence));
        CSVWriter writer_fluency = new CSVWriter(new FileWriter(csv_fluency));
        CSVWriter writer_context_understanding = new CSVWriter(new FileWriter(csv_context_understanding));
        CSVWriter writer_relevance = new CSVWriter(new FileWriter(csv_relevance));


        String[] header = {"Reviewer_1", "Reviewer_2","Reviewer_3"};
        writer_overall_quality.writeNext(header);
        writer_coherence.writeNext(header);
        writer_fluency.writeNext(header);
        writer_context_understanding.writeNext(header);
        writer_relevance.writeNext(header);

        for (Object o : jsonArray) {
            JSONObject jsonObject = (JSONObject) o;
            JSONObject evaluation= (JSONObject) jsonObject.get("evaluation");
            JSONObject human_eval=(JSONObject) evaluation.get("human_eval");
            JSONObject fluency=(JSONObject) human_eval.get("fluency");
            JSONObject coherence=(JSONObject) human_eval.get("coherence");
            JSONObject relevance=(JSONObject) human_eval.get("relevance");
            JSONObject context_understanding=(JSONObject) human_eval.get("context_understanding");
            JSONObject overall_quality=(JSONObject) human_eval.get("overall_quality");


            for(Object value: overall_quality.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_overall_quality,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: coherence.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_coherence,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: fluency.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_fluency,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: relevance.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_relevance,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: context_understanding.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_context_understanding,values.get(0),values.get(1),values.get(2));
            values.clear();



        }

        writer_overall_quality.flush();
        writer_overall_quality.close();
        writer_coherence.flush();
        writer_coherence.close();
        writer_fluency.flush();
        writer_fluency.close();
        writer_relevance.flush();
        writer_relevance.close();
        writer_context_understanding.flush();
        writer_context_understanding.close();


    }

    public static void zephyr_1() throws IOException, ParseException {
        List<Integer> values=new ArrayList<>();
        JSONParser parser = new JSONParser();
        Object obj = parser.parse(new FileReader("ProjektR/test_results_new_dummy/answers/zephyr-7b-alpha.Q5_K_M.gguf.json"));
        JSONArray jsonArray = (JSONArray) obj;


        String csv_overall_quality = "ProjektR/test_results_new_dummy/kappa_zephyir_1/reviewer_scores_overall_quality.csv";
        String csv_coherence = "ProjektR/test_results_new_dummy/kappa_zephyir_1/reviewer_scores_coherence.csv";
        String csv_fluency = "ProjektR/test_results_new_dummy/kappa_zephyir_1/reviewer_scores_fluency.csv";
        String csv_context_understanding = "ProjektR/test_results_new_dummy/kappa_zephyir_1/reviewer_scores_context_understanding.csv";
        String csv_relevance = "ProjektR/test_results_new_dummy/kappa_zephyir_1/reviewer_scores_relevance.csv";

        CSVWriter writer_overall_quality = new CSVWriter(new FileWriter(csv_overall_quality));
        CSVWriter writer_coherence = new CSVWriter(new FileWriter(csv_coherence));
        CSVWriter writer_fluency = new CSVWriter(new FileWriter(csv_fluency));
        CSVWriter writer_context_understanding = new CSVWriter(new FileWriter(csv_context_understanding));
        CSVWriter writer_relevance = new CSVWriter(new FileWriter(csv_relevance));


        String[] header = {"Reviewer_1", "Reviewer_2","Reviewer_3"};
        writer_overall_quality.writeNext(header);
        writer_coherence.writeNext(header);
        writer_fluency.writeNext(header);
        writer_context_understanding.writeNext(header);
        writer_relevance.writeNext(header);

        for (Object o : jsonArray) {
            JSONObject jsonObject = (JSONObject) o;
            JSONObject evaluation= (JSONObject) jsonObject.get("evaluation");
            JSONObject human_eval=(JSONObject) evaluation.get("human_eval");
            JSONObject fluency=(JSONObject) human_eval.get("fluency");
            JSONObject coherence=(JSONObject) human_eval.get("coherence");
            JSONObject relevance=(JSONObject) human_eval.get("relevance");
            JSONObject context_understanding=(JSONObject) human_eval.get("context_understanding");
            JSONObject overall_quality=(JSONObject) human_eval.get("overall_quality");


            for(Object value: overall_quality.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_overall_quality,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: coherence.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_coherence,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: fluency.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_fluency,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: relevance.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_relevance,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: context_understanding.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_context_understanding,values.get(0),values.get(1),values.get(2));
            values.clear();


        }

        writer_overall_quality.flush();
        writer_overall_quality.close();
        writer_coherence.flush();
        writer_coherence.close();
        writer_fluency.flush();
        writer_fluency.close();
        writer_relevance.flush();
        writer_relevance.close();
        writer_context_understanding.flush();
        writer_context_understanding.close();


    }

    public static void zephyr_2() throws IOException, ParseException {
        List<Integer> values=new ArrayList<>();
        JSONParser parser = new JSONParser();
        Object obj = parser.parse(new FileReader("ProjektR/test_results_new_dummy/answers/zephyr-7b-beta.Q5_K_M.gguf.json"));
        JSONArray jsonArray = (JSONArray) obj;


        String csv_overall_quality = "ProjektR/test_results_new_dummy/kappa_zephyir_2/reviewer_scores_overall_quality.csv";
        String csv_coherence = "ProjektR/test_results_new_dummy/kappa_zephyir_2/reviewer_scores_coherence.csv";
        String csv_fluency = "ProjektR/test_results_new_dummy/kappa_zephyir_2/reviewer_scores_fluency.csv";
        String csv_context_understanding = "ProjektR/test_results_new_dummy/kappa_zephyir_2/reviewer_scores_context_understanding.csv";
        String csv_relevance = "ProjektR/test_results_new_dummy/kappa_zephyir_2/reviewer_scores_relevance.csv";

        CSVWriter writer_overall_quality = new CSVWriter(new FileWriter(csv_overall_quality));
        CSVWriter writer_coherence = new CSVWriter(new FileWriter(csv_coherence));
        CSVWriter writer_fluency = new CSVWriter(new FileWriter(csv_fluency));
        CSVWriter writer_context_understanding = new CSVWriter(new FileWriter(csv_context_understanding));
        CSVWriter writer_relevance = new CSVWriter(new FileWriter(csv_relevance));


        String[] header = {"Reviewer_1", "Reviewer_2","Reviewer_3"};
        writer_overall_quality.writeNext(header);
        writer_coherence.writeNext(header);
        writer_fluency.writeNext(header);
        writer_context_understanding.writeNext(header);
        writer_relevance.writeNext(header);

        for (Object o : jsonArray) {
            JSONObject jsonObject = (JSONObject) o;
            JSONObject evaluation= (JSONObject) jsonObject.get("evaluation");
            JSONObject human_eval=(JSONObject) evaluation.get("human_eval");
            JSONObject fluency=(JSONObject) human_eval.get("fluency");
            JSONObject coherence=(JSONObject) human_eval.get("coherence");
            JSONObject relevance=(JSONObject) human_eval.get("relevance");
            JSONObject context_understanding=(JSONObject) human_eval.get("context_understanding");
            JSONObject overall_quality=(JSONObject) human_eval.get("overall_quality");


            for(Object value: overall_quality.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_overall_quality,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: coherence.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_coherence,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: fluency.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_fluency,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: relevance.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_relevance,values.get(0),values.get(1),values.get(2));
            values.clear();


            for(Object value: context_understanding.values()){
                values.add(Math.toIntExact((Long) value));
            }
            writeRow(writer_context_understanding,values.get(0),values.get(1),values.get(2));
            values.clear();


        }

        writer_overall_quality.flush();
        writer_overall_quality.close();
        writer_coherence.flush();
        writer_coherence.close();
        writer_fluency.flush();
        writer_fluency.close();
        writer_relevance.flush();
        writer_relevance.close();
        writer_context_understanding.flush();
        writer_context_understanding.close();


    }

}
