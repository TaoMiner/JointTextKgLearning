package test;

import java.awt.print.Printable;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import Common.Constant;
import EntIndex.AhoCorasickDoubleArrayTrie;
import EntIndex.EntPreProcess;
import EntIndex.IndexBuilder;
import model.Entity;

public class test {

	public static void getMentionList(String input_path, String output_path, int min_count)throws Exception{
		BufferedReader reader = new BufferedReader(new FileReader(input_path));
        BufferedWriter writer = new BufferedWriter(new FileWriter(output_path, false));
        
        String line = null;
        String[] items = null;
        int line_count = 0;
        int sum = 0;
        String tmp = null;
        while((line = reader.readLine())!=null){
        	items = line.split("\t");
        	if(items.length<3)
        		continue;
        	else{
        		sum = 0;
        		for(int i=2;i<items.length;i++){
        			tmp = items[i].replaceAll("(.*)=", "");
        			sum += Integer.parseInt(tmp);
				if(sum>=min_count) break;
        		}
        		if(sum >= min_count){
				line_count++;
	        		for(int i=2;i<items.length;i++){
	        			items[i] = items[i].replaceAll("=[\\d]+", "");
	        			writer.write(items[i]+"::="+items[0]+"\n");
	        		}
			}
        	}
        }
        reader.close();
        writer.close();
	System.out.println("Successfully load "+line_count+" entities.");
	}
	
	public static List<Integer> findMention(List<AhoCorasickDoubleArrayTrie<String>.Hit<String>> mentionList, int doc_lenth){
    	int[] sent = new int[doc_lenth];
    	List<Integer> re_mentions = new ArrayList<>();
    	for(int i=0;i<doc_lenth;i++)
    		sent[i] = -1;
    	AhoCorasickDoubleArrayTrie<String>.Hit<String> tmp_hit = null;
    	for(int i=0;i<mentionList.size();i++){
    		tmp_hit = mentionList.get(i);
    		//if the mention's (begin-1) and (end+1) are not white space or the sentence begin or end, ignore it
//    		if(tmp_hit.begin-1>=0 && doc.charAt(tmp_hit.begin-1)!=' ') continue;
//    		if(tmp_hit.end<doc_lenth && doc.charAt(tmp_hit.end)!=' ') continue;
    		int last_begin_index = sent[tmp_hit.begin];
    		int last_end_index = sent[tmp_hit.end-1];
    		//if both equal -1, then update; or equal other number, then do nothing
    		if(last_begin_index == -1 && last_end_index == -1){
    			for(int j=tmp_hit.begin;j<tmp_hit.end;j++)
    				sent[j] = i;
				continue;
    		}
    		else if(last_begin_index!=last_end_index){
    			int length_begin = last_begin_index==-1 ? 0:(mentionList.get(last_begin_index).end - mentionList.get(last_begin_index).begin);
    			int length_end = last_end_index==-1 ? 0:(mentionList.get(last_end_index).end - mentionList.get(last_end_index).begin);
    			int length_cur = tmp_hit.end - tmp_hit.begin;
    			if(length_cur>length_begin && length_cur>length_end){
    				for(int j=tmp_hit.begin;j<tmp_hit.end;j++)
        				sent[j] = i;
    				if(length_begin!=0)
    					for(int j=mentionList.get(last_begin_index).begin;j<tmp_hit.begin;j++)
            				sent[j] = -1;
    				if(length_end!=0)
    					for(int j=tmp_hit.end;j<mentionList.get(last_end_index).end;j++)
            				sent[j] = -1;
    			}
    		}
        }
    	int cur_index = -1;
    	for(int i=0;i<doc_lenth;i++){
    		if(sent[i]!=-1 && sent[i]!=cur_index){
    			re_mentions.add(sent[i]);
    			cur_index = sent[i];
    		}
    	}
    	return re_mentions;
    }
	
	public static HashMap<String, Integer> countMention(String doc_path, IndexBuilder ibd) throws Exception{
		BufferedReader reader = new BufferedReader(new FileReader(doc_path));
    	String line = null;
    	String label = null;
    	StringBuffer sb = new StringBuffer();
    	List<AhoCorasickDoubleArrayTrie<String>.Hit<String>> can_mention_list = null;
    	Pattern p=Pattern.compile("\\[\\[(.*?)\\]\\]");
    	Matcher m=null;
    	List<Integer> mention_list = null;
    	HashMap<String, Integer> mention_count = new HashMap<>();
    	String tmp_str = null;
	int line_count=0;
    	while((line = reader.readLine())!=null){
		line_count++;
		if(line_count%10000==0) System.out.println("has processed "+line_count+" lines.");
    		int start_pos = 0;
        	int end_pos = 0;
    		m = p.matcher(line);
    		while(m.find()){
    			end_pos = m.start();
    			tmp_str = line.substring(start_pos, end_pos);
    			can_mention_list = ibd.parseText(tmp_str);
    			mention_list = findMention(can_mention_list, end_pos-start_pos);
    			for(Integer index:mention_list){
    				label = tmp_str.substring(can_mention_list.get(index).begin, can_mention_list.get(index).end);
					if(mention_count.containsKey(label))
						mention_count.put(label, mention_count.get(label)+1);
					else
						mention_count.put(label, 1);
    			}
    			//add text mention count
    			start_pos = m.end();
    		}
    		end_pos = line.length();
    		if(start_pos!=end_pos){
    			tmp_str = line.substring(start_pos, end_pos);
    			can_mention_list = ibd.parseText(tmp_str);
    			mention_list = findMention(can_mention_list, end_pos-start_pos);
    			for(Integer index:mention_list){
    				label = tmp_str.substring(can_mention_list.get(index).begin, can_mention_list.get(index).end);

					if(mention_count.containsKey(label))
						mention_count.put(label, mention_count.get(label)+1);
					else
						mention_count.put(label, 1);
    			}
    		}
    		label = null;
    		can_mention_list = null;
    		mention_list = null;
    		m = null;
    		line = null;
    	}
    	
    	return mention_count;
	}
	
	public static void saveCount(String count_path, HashMap<String, Integer> mention_count) throws Exception{
	    BufferedWriter writer = new BufferedWriter(new FileWriter(count_path, false));
	    for (Map.Entry<String, Integer> entry : mention_count.entrySet()) {
	    	writer.write(entry.getKey() + "=" + entry.getValue()+"\n");
        } 
	    writer.close();
	}
	
	public static void main(String[] args) throws Exception{
		String path = "/home/caoyx/JTextKgForEL/data/wiki2016/";
		String entity_path = path+"ent_mention";
		String output_path = "./etc/mention_list";
		String sorted_mention_path = "./etc/mention_list_cl";
		String trie_path = "./etc/mention_list_trie";
		String doc_path = path+"train_text";
		String count_path = path+"mention_count";
		int min_count = 5;
		
		getMentionList(entity_path, output_path, min_count);
		System.out.println("#sorted entities: "+EntPreProcess.EntSort(output_path, sorted_mention_path));
		IndexBuilder ibd = new IndexBuilder();
	        ibd.build(sorted_mention_path);
   	        System.out.println("build trie finished! saving...");
	        ibd.save(trie_path);
//		System.out.println("load trie ...");
//		ibd.load(trie_path);
	    
	    HashMap<String, Integer> mention_count = countMention(doc_path, ibd);
	    saveCount(count_path, mention_count);
	}
	
}
