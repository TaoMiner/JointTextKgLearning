package preprocess;
import Common.Constant;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;

import java.io.IOException;
import java.security.KeyStore.Entry;
import java.util.*;

/**
 * Created by ethan on 16/3/23.
 * Sort the entity-uri list,
 */
public class EntPreProcess {

    //format: <13> property:hasMention "Salavan (city)" .
    //salavan::=13::;city
    public static void formatter(String input_path, String output_path) throws Exception{
        BufferedReader reader = new BufferedReader(new FileReader(input_path));
        BufferedWriter writer = new BufferedWriter(new FileWriter(output_path, false));
        String line=null;
        String[] items = null;
        String uri = null;
        String label = null;
        int count=0;
        while(count<11){
            count++;
            line = reader.readLine();
        }
        count = 0;
        int max_len = 0;
        while ((line = reader.readLine()) != null) {
            if(count%1000000==0)
                System.out.println("has format:"+count);
            count++;
            items = line.split(" ", 3);
            //uri
            uri = items[0].replaceAll("<|>","");
            //label
            label = items[2].replaceAll("\"|\\.","");
            label = label.toLowerCase();
            //split label and description
            String[] tmp = label.split("\\(");
            tmp[0] = tmp[0].trim();
            if(tmp[0].equals(""))
                continue;
            if(label.length()>max_len)
                max_len = label.length();
            if(tmp.length>=2) {
                tmp[1] = tmp[1].replaceAll("\\)", "");
                tmp[1] = tmp[1].trim();
                writer.write(tmp[0]+"::="+uri+"::;"+tmp[1]);
            }
            else {
                writer.write(tmp[0] + "::=" + uri);
            }
            writer.newLine();
        }
        reader.close();
        writer.close();
        System.out.println(max_len);
    }

    public static void getMentionList(String input_path, String output_path, int min_count)throws Exception{
		BufferedReader reader = new BufferedReader(new FileReader(input_path));
        BufferedWriter writer = new BufferedWriter(new FileWriter(output_path, false));
        
        String line = null;
        String[] items = null;
        int line_count = 0;
        int sum = 0;
        String[] tmp = null;
        StringBuffer line_buffer = new StringBuffer();
        while((line = reader.readLine())!=null){
        	items = line.split("\t");
        	if(items.length<3)
        		continue;
        	else{
        		sum = 0;
        		for(int i=2;i<items.length;i++){
        			tmp = items[i].split("=(?=[\\d]+$)");
        			if(tmp.length!=2)
        				continue;
        			sum += Integer.parseInt(tmp[1]);
        			line_buffer.append(tmp[0]).append("::=").append(items[0]).append("\n");
        		}
        		if(sum >= min_count){
        			line_count++;
	        			writer.write(line_buffer.toString());
        		}
        	}
        	line_buffer.setLength(0);
        }
        reader.close();
        writer.close();
	System.out.println("Successfully load "+line_count+" entities.");
	}
    
    public static void main(String[] args) throws Exception{
    	String input_file = null;
    	String output_file = null;
    	int min_count = 1;
    	for(int i=0;i<args.length;i++){
            if(args[i].equals("-input_file"))
            	input_file=args[i+1];
            if(args[i].equals("-output_file"))
            	output_file=args[i+1];
            if(args[i].equals("-min_count"))
            	min_count=Integer.parseInt(args[i+1]);
    	}
    	if(input_file!=null && output_file !=null){
    		getMentionList(input_file,output_file,min_count);
    	}
    }
}
