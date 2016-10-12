package EntIndex;
import Common.Constant;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;

import java.io.IOException;
import java.security.KeyStore.Entry;
import java.util.*;

import model.Entity;

/**
 * Created by ethan on 16/3/23.
 * Sort the entity-uri list,
 */
public class EntPreProcess {

    //format: <13> property:hasMention "Salavan (city)" .
    //salavan::=13::;city
	//格式化大小写为小写
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
            //提取<13> property:hasMention "Salavan (city)" .
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

    //sort the entity list, 不能有重复
    //将相同label的entity合并为一行
    public static int EntSort(String input_path, String output_path)throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(input_path));
        String line;
        Constant.entities = new HashMap<String, String>();
        String[] items;
        int line_count = 0;
        while ((line = reader.readLine()) != null) {
            items = line.split("::=", 2);
            if(items.length<2)
            	continue;
            if(!Constant.entities.containsKey(items[0]))
            	Constant.entities.put(items[0], items[1]);
            else{
            	String tmp_uris = Constant.entities.get(items[0]);
            	String[] uri_des = items[1].split("::;");
            	if(!tmp_uris.matches(uri_des[0]))
            		Constant.entities.put(items[0], tmp_uris+"::="+items[1]);
            }
        }
        reader.close();
        System.out.println("read entities finish! sorting...");
        // sort the entity list
        List<Entity> l_entities = new ArrayList<Entity>();
        for(Map.Entry<String, String> entry:Constant.entities.entrySet())
        	l_entities.add(new Entity(entry.getKey(), entry.getValue()));
        Collections.sort(l_entities);
        System.out.println("sort entities finish! outputting...");
        line_count=0;
        BufferedWriter writer = new BufferedWriter(new FileWriter(output_path, false));
        for (Entity en : l_entities)
        {
            writer.write(en.getLabel() + "::=" + en.getUri());
            writer.newLine();
        }
        writer.close();
        return l_entities.size();
    }
    
    //filter non-[0-9a-zA-Z\u4e00-\u9fa5]
    //length>1
    public static void filter(String input_path, String output_path) throws Exception{
    	BufferedReader reader = new BufferedReader(new FileReader(input_path));
        BufferedWriter writer = new BufferedWriter(new FileWriter(output_path, false));
        String line = null;
        String[] items = null;
        char[] char_line = null;
        HashSet<Character> chars = new HashSet<>();
        int line_count = 0;
        while((line = reader.readLine())!=null){
        	line = line.toLowerCase();
        	items = line.split("::=",2);
        	if(!items[0].matches("[0-9a-zA-Z\u4e00-\u9fa5]+")||items[0].length()<Constant.entity_min_length)
        		continue;
        	else{
        		line_count++;
        		writer.write(line+"\n");
        	}
        	char_line = items[0].toCharArray();
        	for(char tmp:char_line){
        		if(!chars.contains(tmp))
        			chars.add(tmp);
        	}
        }
        System.out.println(line_count);
        reader.close();
        writer.close();
    }
    
    public static void partition(int num, String input_path, String output_path) throws Exception{
    	BufferedReader reader = new BufferedReader(new FileReader(input_path));
        BufferedWriter writer = new BufferedWriter(new FileWriter(output_path, false));
        int line_count = 0;
        String line=null;
        while((line = reader.readLine()) != null){
        	writer.write(line);
        	writer.newLine();
        	if(line_count>=num)
        		break;
        	line_count++;
        }
        reader.close();
        writer.close();
    }
    
    public static void main(String[] args) throws Exception {
//        formatter("./etc/xlore.instance.mention.ttl",Constant.entity_dic_path);
        System.out.println("#sorted entities: "+EntSort(Constant.entity_dic_path, Constant.entity_dic_sorted));
//    	filter(Constant.entity_dic_sorted, Constant.entity_file);
//    	partition(4000000, Constant.entity_file, Constant.entity_file+"_1");
//    	System.out.println("partition finished!");
    }
}
